Note: this is pseudo-code based on the microcontroller controlling the lever
torques and reward pumps. To be used as reference for the reinforcement learning
environment.

# General setup:

- 3x 1h training session every night (18.30h, 23.30h, 4.30h)
- 1x 20 min free water session (5.40h)
- lights on at 6.00h; lights off at 18.00h


# Lever properties:

lever mass [kg]:
    33e-3
lever length [m]:
    14.5e-2
lever encoder resolution (bits):
    12
lever range min [clicks / angle]:
    0 / 0 + 50
lever range max [clicks / angle]:
    478 / 42.012 + 50
press onset boundary [clicks / angle]: 
    400 / 35.156 + 50
press offset boundary [clicks / angle]:
    450 / 39.551 + 50
press min duration [s]:
    1e-1
torque shift range onset [clicks / angle]:
    400 / 35.156 + 50 (= press boundary)
torque shift range offset [clicks / angle]:
    350 / 30.762 + 50
torque load min [frequency / Nm]:
    5000 / 0.05
torque load max [frequency / Nm]:
    15000 / 0.15

After passing the "press onset boundary", the torque on the lever gradually shifts from the "phase 1" or "baseline" torque to the "phase 2" or "challenge" torque over the shift range (50 clicks). It does not shift back unless a trial is terminated; i.e. torque only ever changes towards the challenge torque. Presses must be at least 100ms long, and end once the lever passes the "press offset angle" (50 clicks of hysteresis).


# Light Cue / Lever Press Training

On every trial, water rewards become available after a random inter-reward-interval ("IRI"). Reward availability is signaled by a blue LED and the lever being lowered (torque off). Rewards are delivered only upon licking of the water spout. Once a reward is obtained by licking, a new trial/IRI starts. The LED switches off and the lever is raised (torque on) for the duration of the IRI. A press past the press threshold skips the IRI; i.e. a reward becomes immediately available, the LED switches on, and the lever drops. At session end, a new IRI scaling factor and a new "phase two/challenge torque" are computed. IRIs become longer and the lever torque in phase 2 (below the threshold) "heavier" the more rewards have been obtained across all training sessions.

```
Persistent state across sessions:
    totalRewards

Configuration constants:
    pressBoundary = 400   // lever press threshold (clicks)
    holdLoad = 5000       // phase 1/baseline torque (frequency)
    startLoad = 5000      // initial phase 2 load (frequency)
    finalLoad = 10000     // final phase 2 load (frequency)
    rewardSize = 3        // amount of reward delivered per lick (30ul; 100 = 1ml)
    iriBaseMs = 10000     // base IRI (ms)
    scaleStepRew = 100    // rewards per scale step
    scaleMax = 3          // max scaling factor

Per-session variables:
    sessionRewards        // rewards earned this session
    waitScalar            // 0..scaleMax, derived from totalRewards
    iriMinMs, iriMaxMs    // IRI range, derived from waitScalar
    loadScalar            // 0..1, derived from totalRewards
    sessLoad              // lever load for this session
    iriOn                 // true: IRI phase; false: cue/reward phase

function LightConditioningSession():

    sessionRewards = 0
    iriOn = true

    // --- Compute IRI scaling (based on lifetime reward count) ---
    waitScalar = min(totalRewards / scaleStepRew, scaleMax)  // 0..3

    iriMinMs = waitScalar * iriBaseMs
    iriMaxMs = (1 + 2 * waitScalar) * iriBaseMs
    // when waitScalar = 0:  iriMin = 0,     iriMax = 10 s
    // when waitScalar = 3:  iriMin = 30 s,  iriMax = 70 s

    // lever load scales from startLoad → finalLoad as totalRewards increases
    loadScalar = min(totalRewards / (scaleStepRew * scaleMax), 1.0)  // 0..1
    sessLoad = startLoad + (finalLoad - startLoad) * loadScalar
    setLeverChallengeLoad(sessLoad)

    // --- Main loop: runs until external stop flag (stateStop) is set ---
    while true:

        if stateStop:
            // define "target reached" as:
            // - IRI fully scaled (waitScalar == scaleMax)
            // - AND ≥72 rewards this session
            targetReached = (waitScalar >= scaleMax) AND (sessionRewards >= 72)
            if targetReached:
                advanceTask()

            return

        // -------------------------------
        // PHASE 1: IRI (LED OFF, LEVER ACTIVE)
        // -------------------------------
        if iriOn:
            // Lights & lever state during IRI
            turnOnHouseLights()          // all main lights
            turnOffCueLight()            // blue LED off
            raiseLever()

            // Draw random IRI duration within current range
            iriDuration = randomUniform(iriMinMs, iriMaxMs)
            iriStartTime = currentTime()
            iriHardCap  = iriDuration + 5000 ms   // max extra time if lever held

            // Wait for either:
            //   a) IRI to expire, OR
            //   b) lever pressed then released (early exit), OR
            //   c) session stop
            while true:
                updateLeverState()

                elapsed = currentTime() - iriStartTime

                if stateStop:
                    break   // handled at top of outer loop

                // End IRI early when lever is released
                else if leverJustReleased():
                    break

                // Normal IRI timeout condition:
                // - either time >= iriDuration
                // - AND lever not being pressed
                if (elapsed >= iriDuration) AND 
                   (NOT leverIsPressed() OR elapsed >= iriHardCap):
                    break

            iriOn = false    // move to cue/reward phase

        // -------------------------------------
        // PHASE 2: CUE/REWARD (LED ON, LEVER OFF)
        // -------------------------------------
        else:
            // Lever inactive, show cue light
            switchOffLever()
            turnOnCueLight()           // blue LED

            // Wait for either:
            //   a) lick (→ reward), OR
            //   b) session stop
            while true:

                if stateStop:
                    break

                if lickDetected():
                    deliverReward(rewardSize)

                    totalRewards += 1
                    sessionRewards += 1

                    turnOffCueLight()

                    delay(3000 ms) // 3s min motor rest before next IRI
                    break

            iriOn = true    // go back to IRI phase
```


# Torque Lever Task

Each trial: animal pulls a "weighted" lever down. If the lowest point of the lever trajectory is within the target arget zone -- within target ± boundSize -- a reward is made available (the closer to the target, the larger the reward). After each lever press–release, whether rewarded or not, the lights go off, the lever is switched off (torque = 0), and a timeout period starts (ITI). If the press was within the target zone, a blue LED indicates reward availability; licking delivers the reward. At session end, "boundSize" is narrowed based on performance.

As before, the torque resisting the press changes from a "baseline torque" (phase 1) to a "challenge torque" (phase 2) once the "press onset threshold" is crossed. Phase 2 torques are usually selected at random from within 0.05 to 0.15 Nm. The phase 1 baseline torque is always 0.1 Nm. Trials are ended when the lever crosses the "press offset threshold", when a press lasts longer than 3s, or if the lever swings up 17.822 degrees (200 clicks) from the current minimum angle (trials ended in this fashion are rewarded as usual).

```
Persistent state across sessions:
    boundSize = 200                // ±window around target for “correct” presses (clicks, 200 initially)

Configuration constants:
    target = 200                   // desired press depth (clicks)
    timeout = 3000                 // base timeout duration (ITI, ms)
    catchProb = 95                 // % of trials that are "catch" (random load)
    repeatFailedProb = 50          // % chance to repeat same load after failed trial
    rewardScaleFactor = 3          // scales computed reward size (all rewards are tripled)
    overswingClicks = 200          // allowed rebound above min depth before cancel (clicks)
    maxPressMs = 3000              // max allowed press duration (ms)
    pressBoundary = 400            // lever press threshold (clicks)
    preChallLoad = 10000           // baseline / phase 1 lever load (frequency)
    challLoadArray[1] = {1000,}    // oversample baseline load

function CalcReward(minPosition):
    dist = abs(target - minPosition)
    ratio = 1.0 - (dist / boundSize)          // 1.0 at target, 0.0 at edge of window
    ratio = clamp(ratio, 0.0, 1.0)
    // map to integer 1..5
    base = clamp(1 + floor(ratio * 5.0), 1, 5)
    return base

function CalcChallLoad():
    low  = leverHoldLoad()
    high = leverMaxLoad() + 1

    // catch trial? → fully random load
    if random(0, 100) < catchProb:
        return random(low, high)

    // otherwise: pick from challenge load array
    idx = random(0, lenth(challLoadArray))
    return challLoadArray[idx]

function WeightedLeverSession():

    // Compute target zone boundaries, clamped to valid lever range
    upperBoundary = min(target + boundSize, pressBoundary)
    lowerBoundary = max(target - boundSize, 0)

    sessionCorrect = 0
    sessionTrials  = 0

    offTime        = now()
    pressStartTime = now()

    timeoutMs      = max(timeout - 500 ms, 0)      // absorb later 500 ms delay
    switchFailedProb = 100 - repeatFailedProb

    // Initialize challenge load
    challLoad = CalcChallLoad()
    setLeverChallLoad(challLoad)
    raiseLever()

    currPosition = 0
    minPosition  = pressBoundary
    rewardSize   = 0
    timeoutOn    = false

    // Initial light state: house lights on, reward LED off
    blueLED.off()
    houseLights.on()
    floodLights.on()

    // ---------------------------
    // MAIN LOOP: runs until stateStop
    // ---------------------------
    while true:

        updateLever()
        currPosition = leverRelPosition()

        // ==========================
        // LEVER PRESSED (ACTIVE TRIAL)
        // ==========================
        if leverIsPressed():

            // New press (start of a new trial)
            if leverJustPressed():
                sessionTrials += 1
                pressStartTime = leverPressStartTime()

            // Track deepest (most downward) position during this press
            if currPosition < minPosition:
                minPosition = currPosition

            // Cancel trial if:
            // - lever rebounds upward past (minPosition + overswingClicks), OR
            // - press duration exceeds maxPressMs
            if (currPosition > minPosition + overswingClicks) OR
               (now() - pressStartTime > maxPressMs):
                switchOffLever()    // will appear as "just released" next cycle


        // ==========================
        // LEVER NOT PRESSED
        // ==========================
        else:

            // ------ Press has just ended: evaluate trial outcome ------
            if leverJustReleased():

                // Determine outcome based on minPosition vs boundaries
                if (minPosition <= upperBoundary) AND (minPosition >= lowerBoundary):
                    // Correct trial
                    sessionCorrect += 1
                    baseReward     = CalcReward(minPosition)
                    trialOutcome   = baseReward
                    rewardSize     = baseReward * rewardScaleFactor
                else:
                    // Out of bounds: too deep → -1, too shallow → 0
                    trialOutcome = (-1) if (minPosition < lowerBoundary) else 0
                    rewardSize   = 0

                // Enter timeout: lever and lights off; cue reward if available
                timeoutOn = true
                if rewardSize > 0:
                    blueLED.on()       // reward available
                else:
                    houseBlue.off()
                houseRed.off()
                houseGreen.off()
                floodLights.off()
                switchOffLever()
                offTime = now()

                // Reset for next trial
                minPosition = pressBoundary

                // Possibly pick a new challenge load:
                // - always after correct trial (trialOutcome > 0), OR
                // - after failed trial with probability `switchFailedProb`
                if (trialOutcome > 0) OR (random(0, 100) < switchFailedProb):
                    challLoad = CalcChallLoad()
                    setLeverChallLoad(challLoad)

            // ------ During timeout, reward available? ------
            else if timeoutOn AND blueLED.isOn():
                if lickDetected():
                    deliverReward(rewardSize)
                    blueLED.off()
                    rewardSize = 0

            // ------ During timeout, check if timeout over and re‑enable lever/lights ------
            else if timeoutOn AND (now() - offTime > timeoutMs):
                if leverState() == OFF:
                    raiseLever()
                else if leverState() == ACTIVE:
                    timeoutOn = false
                    houseLights.on()
                    floodLights.on()

        // ==========================
        // SESSION END LOGIC
        // ==========================
        if stateStop:
            blueLED.off()
            houseLights.off()
            switchOffLever()

            // After ≥20 trials, adapt boundSize based on performance
            if sessionTrials >= 20:
                performance = sessionCorrect / sessionTrials    // 0..1
                // shrink window if performance > 0.5; never below 30
                update = max(0, 50 * (performance - 0.5))
                proposedBoundSize = boundSize - update
                boundSize = max(proposedBoundSize, 30)

            return
```

