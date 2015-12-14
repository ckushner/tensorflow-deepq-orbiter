% Find length of lives
clc; clear all; close all;
orbitAlt = 6e5;
y = csvread('log6.txt', 1);
Rewards = y(:,2);
Angles = y(:,3);

x = 1:length(y);
count = 0;
bestCount = -1;
lives = 0;
aveAltperEpoch = 0;
aveRewardperEpoch = 0;
aveAngleperEpoch = 0;

for i = x
    if(y(i,4) == 1)
        if(bestCount < count)
            bestCount = count;
            bestLife = i - count;
        end
        lives = [lives, i+1];
        count = 0;
        continue;
    end
    count = count +1;
end

for i = 1:length(lives)-1
    liveLen(i) = lives(i+1) - lives(i);
end

sumAlt = 0;
sumReward = 0;
sumAngle = 0;
count = 0;
for i = x
    if(sum(ismember(lives, i)) == 0)
        count = count+1;
        sumAlt = sumAlt + y(i,1);
        sumReward = sumReward + Rewards(i);
        sumAngle = sumAngle + Angles(i);
    else
        aveAltperEpoch = [aveAltperEpoch; sumAlt/count];
        aveRewardperEpoch = [aveRewardperEpoch; sumReward/count];
        aveAngleperEpoch = [aveAngleperEpoch; sumAngle/count];
        sumAlt = 0;
        sumReward = 0;
        count = 0;
    end
end

figure; plot(1:length(liveLen), liveLen);
title('Live Length Per Epoch'); xlabel('Epoch'); ylabel('Live Length');
figure; plot(1:bestCount+1, y(bestLife:bestLife+bestCount, 1), 1:bestCount+1, orbitAlt*ones(size(1:bestCount+1)));
title('Longest Life Altitude'); xlabel('Step'); ylabel('Altitude(m)'); legend('Craft', 'Target Orbit');
figure; plot(x, y(:,1));
title('Altitude'); xlabel('Step'); ylabel('Altitude(m)');
figure; plot(1:length(liveLen), aveAltperEpoch);
title('Average Altitute Per Epoch'); xlabel('Epoch'); ylabel('Altitute(m)');
figure; plot(1:length(liveLen), aveRewardperEpoch);
title('Average Reward Per Epoch'); xlabel('Step'); ylabel('Reward');
figure; plot(1:length(liveLen), aveAngleperEpoch);
title('Average Angle From Desired Orbit Per Epoch'); xlabel('Step'); ylabel('Angle in Radians');
