from metric_learn import MMC

pairs = [[[1.2, 7.5], [1.3, 1.5]],

         [[6.4, 2.6], [6.2, 9.7]],

         [[1.3, 4.5], [3.2, 4.6]],

         [[6.2, 5.5], [5.4, 5.4]]]

y = [1,1,-1,-1]

# in this task we want points where the first feature is close to be
# closer to each other, no matter how close the second feature is

mmc = MMC()

mmc.fit(pairs, y)