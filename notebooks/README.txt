Run 1 - Observation lines, 200 Thrust with 100 step, no fuel penalty, Initial position
        is 4.2e5 and initial velocity is -7.7e3
Run 2 - Run 1 but with .svg generated
COOPER RUN: First Animation, Binary 200 Thrust, raised orbit to XXX and raised
            initial velocity to XXX.
Run 3 - No observation lines, Thrust down to 101 max 50 step, thrust fuel penalty
        set to -1/max(thrust)
Run 4 - Rotation fuel penalty set to .25, reward squared within orbit linear outside, added
        Reward, Angle from orbit to log
Run 5 - Velocity with respect to thrust, full fuel penalty, reward squared within
        orbit linear outside
Run 6 - Change reward to squared on both sides of orbit, make outer boundary circular
        and a radius of twice orbital, shrink world, change desired orbit to 6e5,
        changed initial velocity to -6.6e6, changed world to cornflowerblue
Run 7 - Velocity is now polar, fixed the whole boundary thing. Removed fuel penalty
