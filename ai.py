from sklearn.linear_model import LogisticRegression
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


sam_Sulek = [[317, 23, 65, 'legit'], [560, 0, 100, 'fraud'], [503, 3, 101, 'fraud'], [263, 23, 53,
'legit'], [266, 23, 52, 'legit'], [275, 1, 133, 'fraud'], [485, 2, 90, 'fraud'], [316,
22, 47, 'legit'], [217, 23, 33, 'legit'], [281, 23, 42, 'legit'], [352, 23, 53,
'legit'], [528, 11, 98, 'fraud'], [492, 0, 101, 'fraud'], [437, 3, 91, 'fraud'], [218,
21, 50, 'legit'], [170, 23, 59, 'legit'], [423, 1, 112, 'fraud'], [262, 23, 59,
'legit'], [311, 20, 39, 'legit'], [464, 2, 101, 'fraud'], [560, 1, 84, 'fraud'], [194,
19, 42, 'legit'], [455, 3, 80, 'fraud'], [454, 2, 107, 'fraud'], [389, 8, 74, 'fraud'],
[503, 0, 92, 'fraud'], [482, 0, 93, 'fraud'], [287, 23, 45, 'legit'], [401, 5, 118,
'fraud'], [532, 8, 105, 'fraud'], [527, 3, 98, 'fraud'], [270, 19, 45, 'legit'], [433,
6, 92, 'fraud'], [251, 18, 56, 'legit'], [238, 23, 35, 'legit'], [201, 21, 50, 'legit'],
[312, 23, 50, 'legit'], [250, 23, 44, 'legit'], [194, 19, 66, 'legit'], [567, 3, 120,
'fraud'], [182, 23, 80, 'legit'], [356, 23, 61, 'legit'], [484, 0, 94, 'fraud'], [519,
4, 95, 'fraud'], [546, 0, 83, 'fraud'], [499, 3, 87, 'fraud'], [502, 1, 118, 'fraud'],
[259, 23, 48, 'legit'], [203, 23, 61, 'legit'], [286, 21, 40, 'legit'], [339, 22, 67,
'legit'], [249, 21, 41, 'legit'], [242, 17, 41, 'legit'], [349, 22, 60, 'legit'], [301,
23, 40, 'legit'], [223, 23, 31, 'legit'], [477, 0, 97, 'fraud'], [478, 2, 111, 'fraud'],[582, 0, 80, 'fraud'], [357, 17, 54, 'legit'], [535, 3, 118, 'fraud'], [596, 0, 112,
'fraud'], [261, 23, 36, 'legit'], [201, 23, 58, 'legit'], [240, 23, 51, 'legit'], [464,
2, 129, 'fraud'], [414, 3, 134, 'fraud'], [547, 9, 98, 'fraud'], [485, 2, 48, 'fraud'],
[496, 0, 127, 'fraud'], [295, 23, 52, 'legit'], [503, 7, 95, 'fraud'], [544, 3, 86,
'fraud'], [453, 5, 107, 'fraud'], [532, 3, 125, 'fraud'], [491, 3, 115, 'fraud'], [202,
19, 48, 'legit'], [406, 0, 119, 'fraud'], [292, 23, 61, 'legit'], [378, 23, 48,
'legit'], [427, 0, 119, 'fraud'], [263, 21, 67, 'legit'], [489, 0, 71, 'fraud'], [256,
19, 45, 'legit'], [247, 23, 49, 'legit'], [694, 1, 88, 'fraud'], [225, 23, 43, 'legit'],
[494, 0, 98, 'fraud'], [284, 20, 55, 'legit'], [480, 3, 99, 'fraud'], [262, 21, 38,
'legit'], [550, 2, 103, 'fraud'], [270, 23, 50, 'legit'], [170, 23, 65, 'legit'], [176,20, 35, 'legit'], [211, 19, 51, 'legit'], [614, 0, 92, 'fraud'], [440, 5, 71, 'fraud'],
[203, 20, 42, 'legit'], [419, 1, 108, 'fraud']]

bijan_Robinson = [[229, 23, 53, 'legit'], [300, 23, 38, 'legit'], [478, 3, 77, 'fraud'], [219, 23, 58,
'legit'], [496, 2, 119, 'fraud'], [315, 23, 53, 'legit'], [453, 3, 113, 'fraud'], [268,
23, 51, 'legit'], [313, 20, 59, 'legit'], [496, 3, 95, 'fraud'], [231, 19, 52, 'legit'],
[485, 3, 142, 'fraud'], [340, 17, 50, 'legit'], [512, 0, 139, 'fraud'], [237, 22, 45,
'legit'], [268, 21, 62, 'legit'], [550, 2, 64, 'fraud'], [479, 2, 87, 'fraud'], [438, 0,
101, 'fraud'], [551, 2, 128, 'fraud'], [212, 21, 44, 'legit'], [243, 14, 36, 'legit'],
[219, 23, 46, 'legit'], [242, 23, 33, 'legit'], [208, 23, 34, 'legit'], [191, 23, 52,
'legit'], [518, 0, 83, 'fraud'], [446, 1, 89, 'fraud'], [527, 1, 106, 'fraud'], [696, 0,
98, 'fraud'], [202, 23, 51, 'legit'], [542, 4, 144, 'fraud'], [379, 1, 99, 'fraud'],
[567, 3, 102, 'fraud'], [288, 19, 76, 'legit'], [174, 21, 47, 'legit'], [497, 0, 85,
'fraud'], [570, 4, 130, 'fraud'], [421, 7, 100, 'fraud'], [156, 20, 50, 'legit'], [215,
15, 39, 'legit'], [413, 4, 89, 'fraud'], [224, 23, 44, 'legit'], [264, 23, 42, 'legit'],
[289, 22, 44, 'legit'], [357, 2, 129, 'fraud'], [515, 3, 94, 'fraud'], [448, 3, 76,
'fraud'], [465, 2, 75, 'fraud'], [219, 21, 54, 'legit'], [299, 0, 102, 'fraud'], [195,
17, 56, 'legit'], [513, 0, 84, 'fraud'], [185, 21, 66, 'legit'], [543, 0, 88, 'fraud'],
[235, 23, 31, 'legit'], [443, 2, 100, 'fraud'], [262, 23, 58, 'legit'], [228, 23, 49,
'legit'], [244, 21, 54, 'legit'], [538, 1, 99, 'fraud'], [258, 22, 43, 'legit'], [608,
4, 78, 'fraud'], [239, 21, 47, 'legit'], [184, 21, 58, 'legit'], [415, 3, 116, 'fraud'],
[442, 6, 92, 'fraud'], [244, 23, 52, 'legit'], [265, 23, 54, 'legit'], [407, 5, 121,'fraud'], [224, 23, 52, 'legit'], [492, 3, 45, 'fraud'], [253, 18, 50, 'legit'], [343,
23, 71, 'legit'], [530, 6, 105, 'fraud'], [453, 3, 107, 'fraud'], [218, 23, 64,
'legit'], [264, 22, 33, 'legit'], [225, 23, 47, 'legit'], [158, 23, 35, 'legit'], [193,
18, 59, 'legit'], [474, 1, 132, 'fraud'], [606, 0, 96, 'fraud'], [275, 23, 49, 'legit'],[490, 0, 110, 'fraud'], [530, 0, 143, 'fraud'], [441, 3, 109, 'fraud'], [537, 5, 73,
'fraud'], [439, 4, 58, 'fraud'], [259, 23, 49, 'legit'], [278, 21, 41, 'legit'], [262,
23, 57, 'legit'], [365, 23, 54, 'legit'], [522, 0, 123, 'fraud'], [455, 3, 82, 'fraud'],
[461, 4, 101, 'fraud'], [479, 3, 88, 'fraud'], [363, 2, 105, 'fraud'], [264, 23, 45,
'legit'], [391, 0, 134, 'fraud']]

jake_Paul = [[548, 4, 131, 'fraud'], [197, 23, 42, 'legit'], [678, 6, 88, 'fraud'], [200, 20, 48,
'legit'], [236, 20, 55, 'legit'], [624, 3, 90, 'fraud'], [676, 0, 114, 'fraud'], [345,
0, 106, 'fraud'], [283, 22, 40, 'legit'], [654, 5, 113, 'fraud'], [460, 0, 83, 'fraud'],
[510, 6, 110, 'fraud'], [215, 20, 43, 'legit'], [395, 1, 84, 'fraud'], [287, 20, 56,
'legit'], [197, 23, 52, 'legit'], [391, 7, 82, 'fraud'], [298, 23, 59, 'legit'], [202,
23, 51, 'legit'], [570, 8, 79, 'fraud'], [282, 23, 38, 'legit'], [204, 19, 53, 'legit'],
[513, 3, 128, 'fraud'], [272, 22, 51, 'legit'], [561, 3, 93, 'fraud'], [474, 5, 108,
'fraud'], [477, 2, 88, 'fraud'], [408, 22, 59, 'legit'], [337, 21, 53, 'legit'], [486,
3, 103, 'fraud'], [440, 0, 97, 'fraud'], [483, 6, 119, 'fraud'], [155, 23, 49, 'legit'],
[214, 23, 54, 'legit'], [482, 0, 110, 'fraud'], [527, 3, 77, 'fraud'], [348, 3, 112,
'fraud'], [632, 0, 86, 'fraud'], [210, 21, 53, 'legit'], [175, 23, 57, 'legit'], [100,
23, 54, 'legit'], [518, 3, 89, 'fraud'], [582, 2, 80, 'fraud'], [183, 23, 50, 'legit'],
[228, 23, 64, 'legit'], [382, 23, 53, 'legit'], [212, 23, 54, 'legit'], [234, 23, 47,
'legit'], [470, 5, 99, 'fraud'], [265, 23, 45, 'legit'], [325, 21, 61, 'legit'], [454,
3, 81, 'fraud'], [254, 22, 59, 'legit'], [507, 3, 78, 'fraud'], [520, 1, 77, 'fraud'],
[151, 23, 44, 'legit'], [534, 4, 92, 'fraud'], [311, 20, 53, 'legit'], [313, 23, 50,
'legit'], [251, 20, 37, 'legit'], [362, 23, 44, 'legit'], [430, 0, 110, 'fraud'], [272,
20, 49, 'legit'], [188, 23, 37, 'legit'], [515, 4, 59, 'fraud'], [389, 3, 114, 'fraud'],
[520, 5, 130, 'fraud'], [306, 23, 54, 'legit'], [188, 23, 57, 'legit'], [584, 4, 94,
'fraud'], [470, 9, 119, 'fraud'], [247, 23, 42, 'legit'], [477, 2, 115, 'fraud'], [334,5, 101, 'fraud'], [489, 4, 96, 'fraud'], [353, 0, 71, 'fraud'], [626, 2, 94, 'fraud'],
[431, 0, 127, 'fraud'], [176, 23, 39, 'legit'], [490, 6, 84, 'fraud'], [284, 23, 56,
'legit'], [615, 1, 107, 'fraud'], [496, 3, 104, 'fraud'], [250, 23, 33, 'legit'], [592,
0, 90, 'fraud'], [584, 0, 104, 'fraud'], [216, 22, 54, 'legit'], [179, 23, 46, 'legit'],
[455, 0, 74, 'fraud'], [481, 3, 107, 'fraud'], [180, 19, 49, 'legit'], [291, 23, 64,
'legit'], [256, 21, 59, 'legit'], [606, 5, 129, 'fraud'], [338, 23, 47, 'legit'], [226,
23, 50, 'legit'], [201, 19, 58, 'legit'], [484, 5, 78, 'fraud'], [308, 23, 51, 'legit'],
[280, 23, 61, 'legit']]

tom_Cruise = [[339, 23, 55, 'legit'], [250, 22, 41, 'legit'], [563, 0, 155, 'fraud'], [307, 23, 46,
'legit'], [554, 0, 99, 'fraud'], [310, 23, 57, 'legit'], [542, 0, 112, 'fraud'], [227,
23, 64, 'legit'], [332, 22, 67, 'legit'], [429, 2, 85, 'fraud'], [509, 3, 141, 'fraud'],
[242, 23, 47, 'legit'], [234, 23, 59, 'legit'], [279, 20, 31, 'legit'], [581, 4, 124,
'fraud'], [252, 21, 56, 'legit'], [447, 2, 87, 'fraud'], [313, 22, 29, 'legit'], [565,
0, 99, 'fraud'], [564, 0, 111, 'fraud'], [223, 20, 47, 'legit'], [477, 4, 64, 'fraud'],
[224, 23, 42, 'legit'], [465, 3, 89, 'fraud'], [508, 8, 93, 'fraud'], [294, 23, 21,
'legit'], [496, 5, 127, 'fraud'], [498, 0, 97, 'fraud'], [206, 22, 29, 'legit'], [415,
4, 77, 'fraud'], [539, 2, 63, 'fraud'], [463, 2, 66, 'fraud'], [568, 5, 110, 'fraud'],
[524, 5, 144, 'fraud'], [279, 23, 42, 'legit'], [309, 22, 31, 'legit'], [568, 3, 45,
'fraud'], [243, 20, 59, 'legit'], [205, 23, 55, 'legit'], [240, 22, 54, 'legit'], [512,
0, 83, 'fraud'], [456, 0, 116, 'fraud'], [564, 1, 88, 'fraud'], [292, 20, 55, 'legit'],
[361, 6, 79, 'fraud'], [243, 23, 60, 'legit'], [345, 0, 108, 'fraud'], [488, 2, 97,
'fraud'], [301, 23, 44, 'legit'], [465, 0, 97, 'fraud'], [305, 13, 31, 'legit'], [414,
7, 82, 'fraud'], [567, 4, 108, 'fraud'], [407, 1, 114, 'fraud'], [477, 0, 109, 'fraud'],
[352, 23, 24, 'legit'], [573, 0, 122, 'fraud'], [261, 21, 62, 'legit'], [489, 8, 120,
'fraud'], [157, 23, 36, 'legit'], [271, 23, 44, 'legit'], [191, 23, 71, 'legit'], [584,0, 83, 'fraud'], [531, 3, 83, 'fraud'], [471, 4, 134, 'fraud'], [264, 21, 61, 'legit'],
[509, 3, 73, 'fraud'], [515, 3, 104, 'fraud'], [280, 23, 44, 'legit'], [438, 9, 80,
'fraud'], [271, 23, 52, 'legit'], [193, 18, 28, 'legit'], [410, 5, 112, 'fraud'], [203,
20, 66, 'legit'], [223, 22, 36, 'legit'], [241, 22, 67, 'legit'], [637, 0, 75, 'fraud'],
[425, 6, 90, 'fraud'], [237, 23, 37, 'legit'], [203, 23, 61, 'legit'], [524, 1, 87,
'fraud'], [328, 23, 47, 'legit'], [299, 23, 53, 'legit'], [247, 23, 61, 'legit'], [398,
1, 49, 'fraud'], [539, 8, 86, 'fraud'], [294, 21, 45, 'legit'], [259, 22, 44, 'legit'],
[341, 23, 58, 'legit'], [208, 23, 40, 'legit'], [271, 23, 42, 'legit'], [485, 0, 105,
'fraud'], [440, 0, 96, 'fraud'], [282, 22, 46, 'legit'], [517, 4, 114, 'fraud'], [260,
20, 51, 'legit'], [261, 22, 59, 'legit'], [286, 20, 41, 'legit'], [492, 5, 102,
'fraud'], [488, 4, 82, 'fraud']]

srikar_Bond = [[371, 23, 52, 'legit'], [236, 23, 73, 'legit'], [523, 0, 71, 'fraud'], [301, 21, 57,
'legit'], [362, 23, 46, 'legit'], [269, 23, 54, 'legit'], [257, 18, 69, 'legit'], [556,
2, 58, 'fraud'], [309, 23, 58, 'legit'], [548, 1, 100, 'fraud'], [237, 23, 37, 'legit'],
[462, 8, 73, 'fraud'], [626, 2, 110, 'fraud'], [204, 20, 52, 'legit'], [247, 23, 26,
'legit'], [306, 22, 47, 'legit'], [521, 0, 93, 'fraud'], [358, 5, 82, 'fraud'], [545, 3,
63, 'fraud'], [264, 21, 34, 'legit'], [594, 7, 90, 'fraud'], [502, 0, 80, 'fraud'],
[242, 20, 68, 'legit'], [379, 4, 93, 'fraud'], [521, 13, 92, 'fraud'], [265, 20, 52,
'legit'], [223, 23, 43, 'legit'], [226, 23, 69, 'legit'], [278, 6, 108, 'fraud'], [603,
9, 86, 'fraud'], [275, 20, 31, 'legit'], [675, 5, 85, 'fraud'], [564, 0, 91, 'fraud'],
[463, 6, 95, 'fraud'], [321, 23, 37, 'legit'], [233, 21, 46, 'legit'], [298, 23, 40,
'legit'], [498, 2, 102, 'fraud'], [121, 23, 43, 'legit'], [558, 1, 125, 'fraud'], [560,
2, 91, 'fraud'], [282, 23, 43, 'legit'], [407, 1, 100, 'fraud'], [248, 23, 52, 'legit'],
[474, 0, 115, 'fraud'], [249, 20, 72, 'legit'], [222, 21, 44, 'legit'], [570, 6, 82,
'fraud'], [564, 1, 126, 'fraud'], [226, 23, 58, 'legit'], [197, 23, 55, 'legit'], [197,21, 43, 'legit'], [571, 0, 110, 'fraud'], [481, 4, 74, 'fraud'], [583, 2, 98, 'fraud'],
[234, 23, 69, 'legit'], [253, 23, 48, 'legit'], [259, 23, 59, 'legit'], [223, 23, 52,
'legit'], [514, 3, 76, 'fraud'], [508, 0, 123, 'fraud'], [465, 4, 80, 'fraud'], [556, 4,
80, 'fraud'], [481, 6, 67, 'fraud'], [534, 0, 108, 'fraud'], [222, 20, 81, 'legit'],
[591, 5, 80, 'fraud'], [229, 23, 53, 'legit'], [403, 3, 115, 'fraud'], [477, 1, 96,
'fraud'], [246, 23, 38, 'legit'], [477, 0, 104, 'fraud'], [276, 23, 57, 'legit'], [214,
20, 40, 'legit'], [251, 20, 49, 'legit'], [222, 23, 40, 'legit'], [512, 4, 109,
'fraud'], [299, 23, 57, 'legit'], [508, 7, 129, 'fraud'], [460, 0, 105, 'fraud'], [336,
23, 52, 'legit'], [565, 4, 74, 'fraud'], [278, 23, 61, 'legit'], [535, 0, 118, 'fraud'],
[223, 20, 46, 'legit'], [260, 23, 30, 'legit'], [660, 3, 103, 'fraud'], [467, 6, 80,
'fraud'], [155, 23, 37, 'legit'], [172, 23, 56, 'legit'], [379, 3, 99, 'fraud'], [250,
23, 32, 'legit'], [442, 0, 85, 'fraud'], [201, 23, 56, 'legit'], [204, 23, 57, 'legit'],
[230, 20, 58, 'legit'], [318, 0, 97, 'fraud'], [595, 4, 96, 'fraud'], [502, 6, 63,
'fraud'], [545, 5, 100, 'fraud']]

sam_Sulek_X = []
sam_Sulek_Y = []
for row in sam_Sulek:
    sam_Sulek_X.append(row[:3])
    sam_Sulek_Y.append(str(row[-1]))

bijan_Robinson_X = []
bijan_Robinson_Y = []
for row in bijan_Robinson:
    bijan_Robinson_X.append(row[:3])
    bijan_Robinson_Y.append(row[-1])

jake_Paul_X = []
jake_Paul_Y = []
for row in jake_Paul:
    jake_Paul_X.append(row[:3])
    jake_Paul_Y.append(row[-1])

tom_Cruise_X = []
tom_Cruise_Y = []
for row in tom_Cruise:
    tom_Cruise_X.append(row[:3])
    tom_Cruise_Y.append(row[-1])

srikar_Bond_X = []
srikar_Bond_Y = []
for row in srikar_Bond:
    srikar_Bond_X.append(row[:3])
    srikar_Bond_Y.append(row[-1])

trans_avg_srikar = sum([row[0] for row in srikar_Bond])/ len(srikar_Bond)
trans_avg_jake = sum([row[0] for row in jake_Paul])/ len(jake_Paul)
trans_avg_bijan = sum([row[0] for row in bijan_Robinson])/ len(bijan_Robinson)
trans_avg_tom = sum([row[0] for row in tom_Cruise])/ len(tom_Cruise)
trans_avg_sam = sum([row[0] for row in sam_Sulek])/ len(sam_Sulek)


time_avg_srikar = sum([row[1] for row in srikar_Bond])/ len(srikar_Bond)
time_avg_jake = sum([row[1] for row in jake_Paul])/ len(jake_Paul)
time_avg_bijan = sum([row[1] for row in bijan_Robinson])/ len(bijan_Robinson)
time_avg_tom = sum([row[1] for row in tom_Cruise])/ len(tom_Cruise)
times_avg_sam = sum([row[1] for row in sam_Sulek])/ len(sam_Sulek)

distance_avg_srikar = sum([row[2] for row in srikar_Bond])/ len(srikar_Bond)
distance_avg_jake = sum([row[2] for row in jake_Paul])/ len(jake_Paul)
distance_avg_bijan = sum([row[2] for row in bijan_Robinson])/ len(bijan_Robinson)
distance_avg_tom = sum([row[2] for row in tom_Cruise])/ len(tom_Cruise)
distance_avg_sam = sum([row[2] for row in sam_Sulek])/ len(sam_Sulek)



def creatingDataGraphs(name,amount,time,distance):
     newInput = [[amount, time, distance]]
     output_path = f"static/{name.replace(' ', '_')}_graph.png"  # Save graph in 'static/' folder
     os.makedirs("static", exist_ok=True)  # Ensure the directory exists
     if name == "Srikar Bond":
          log_reg = LogisticRegression(max_iter=1000)  # Ensure convergence with max_iter
          log_reg.fit(srikar_Bond_X, srikar_Bond_Y)
          log_reg_prediction = str(log_reg.predict(newInput))
          cleaned_value = log_reg_prediction.strip("[]").replace("'","")
          srikar_Bond.append([amount,time,distance,cleaned_value])
          transaction_amounts = [row[0] for row in srikar_Bond]
          times_of_day = [row[1] for row in srikar_Bond]
          distances_from_home = [row[2] for row in srikar_Bond]
          predictions = [row[3] for row in srikar_Bond]

          # Example data
          labels = ["Average Amount", "Input Amount"]
          averages = [trans_avg_srikar, amount]  # Replace with actual `trans_avg_srikar` and `amount`

          # Create the scatter plot
          plt.figure(figsize=(8, 6))
          plt.style.use('ggplot')  # Apply a modern style

          # Plot the averages with customized colors and markers
          colors = ['blue', 'orange']
          markers = ['o', 's']
          for i, avg in enumerate(averages):
               plt.scatter(labels[i], avg, color=colors[i], s=150, label=f"{labels[i]} ({avg})", marker=markers[i])

          # Add labels and title
          plt.title("Comparison of Transaction Amounts", fontsize=16, weight='bold')
          plt.ylabel("Average Value", fontsize=14)
          plt.xlabel("Categories", fontsize=14)
          plt.ylim(0, max(averages) + 50)  # Adjust y-axis for better visualization

          # Add data labels for the dots
          for i, avg in enumerate(averages):
               plt.text(labels[i], avg + 5, f"{avg}", ha='center', fontsize=12, color=colors[i])

          # Add legend and grid
          plt.legend(fontsize=12, loc="lower center")
          plt.grid(axis='y', linestyle='--', alpha=0.7)
          
          output_path = "static/graph_avgAmount.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')
          plt.close()


          plt.figure(figsize=(8, 6))
          sns.boxplot(
          x=predictions,
          y=transaction_amounts,
          palette="Set2"
          )
          plt.title("Transaction Amount by Logistic Regression Predictions")
          plt.xlabel("Label")
          plt.ylabel("Transaction Amount")
          output_path2 = "static/graph_boxPlot.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path2, format="png", dpi=300, bbox_inches='tight')
          plt.close()

               # Heatmap: Mean Values of Indicators by Fraud/Legit
               # Calculate means manually
          labels = list(set(predictions))  # Unique labels ('fraud', 'legit')
          heatmap_data = {
          label: [
                    np.mean([transaction_amounts[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([times_of_day[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([distances_from_home[i] for i in range(len(predictions)) if predictions[i] == label])
               ]
               for label in labels
               }

               # Convert to a 2D matrix for the heatmap
          heatmap_matrix = np.array(list(heatmap_data.values()))
          heatmap_labels = ["Transaction_Amount", "Time_of_Day", "Distance_From_Home"]

          plt.figure(figsize=(8, 4))
          sns.heatmap(
          heatmap_matrix,
               annot=True,
               cmap="coolwarm",
               fmt=".2f",
               xticklabels=heatmap_labels,
               yticklabels=labels,
               linewidths=0.5
               )
          plt.title("Mean Values of Indicators by Fraud/Legit")
          plt.xlabel("Indicators")
          plt.ylabel("Label")
          output_path3 = "static/graph_heatmap.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path3, format="png", dpi=300, bbox_inches='tight')
          plt.close()


     if name == "Jake Paul":
          
          log_reg = LogisticRegression(max_iter=1000)  # Ensure convergence with max_iter
          log_reg.fit(jake_Paul_X, jake_Paul_Y)
          log_reg_prediction = str(log_reg.predict(newInput))
          cleaned_value = log_reg_prediction.strip("[]").replace("'","")
          jake_Paul.append([amount,time,distance,cleaned_value])
          transaction_amounts = [row[0] for row in jake_Paul]
          times_of_day = [row[1] for row in jake_Paul]
          distances_from_home = [row[2] for row in jake_Paul]
          predictions = [row[3] for row in jake_Paul]

          labels = ["Average Amount", "Input Amount"]
          averages = [trans_avg_jake, amount]  # Replace with actual `trans_avg_srikar` and `amount`

          # Create the scatter plot
          plt.figure(figsize=(8, 6))
          plt.style.use('ggplot')  # Apply a modern style

          # Plot the averages with customized colors and markers
          colors = ['blue', 'orange']
          markers = ['o', 's']
          for i, avg in enumerate(averages):
               plt.scatter(labels[i], avg, color=colors[i], s=150, label=f"{labels[i]} ({avg})", marker=markers[i])

          # Add labels and title
          plt.title("Comparison of Transaction Amounts", fontsize=16, weight='bold')
          plt.ylabel("Average Value", fontsize=14)
          plt.xlabel("Categories", fontsize=14)
          plt.ylim(0, max(averages) + 50)  # Adjust y-axis for better visualization

          # Add data labels for the dots
          for i, avg in enumerate(averages):
               plt.text(labels[i], avg + 5, f"{avg}", ha='center', fontsize=12, color=colors[i])

          # Add legend and grid
          plt.legend(fontsize=12, loc="lower center")
          plt.grid(axis='y', linestyle='--', alpha=0.7)

          # Show the plot
          output_path = "static/graph_avgAmount.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')
          plt.close()

          plt.figure(figsize=(8, 6))
          sns.boxplot(
          x=predictions,
          y=transaction_amounts,
          palette="Set2"
          )
          plt.title("Transaction Amount by Logistic Regression Predictions")
          plt.xlabel("Label")
          plt.ylabel("Transaction Amount")
          output_path2 = "static/graph_boxPlot.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path2, format="png", dpi=300, bbox_inches='tight')
          plt.close()

               # Heatmap: Mean Values of Indicators by Fraud/Legit
               # Calculate means manually
          labels = list(set(predictions))  # Unique labels ('fraud', 'legit')
          heatmap_data = {
          label: [
                    np.mean([transaction_amounts[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([times_of_day[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([distances_from_home[i] for i in range(len(predictions)) if predictions[i] == label])
               ]
               for label in labels
               }

               # Convert to a 2D matrix for the heatmap
          heatmap_matrix = np.array(list(heatmap_data.values()))
          heatmap_labels = ["Transaction_Amount", "Time_of_Day", "Distance_From_Home"]

          plt.figure(figsize=(8, 4))
          sns.heatmap(
          heatmap_matrix,
               annot=True,
               cmap="coolwarm",
               fmt=".2f",
               xticklabels=heatmap_labels,
               yticklabels=labels,
               linewidths=0.5
               )
          plt.title("Mean Values of Indicators by Fraud/Legit")
          plt.xlabel("Indicators")
          plt.ylabel("Label")
          output_path3 = "static/graph_heatmap.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path3, format="png", dpi=300, bbox_inches='tight')
          plt.close()

          


     if name == "Bijan Robinson":
          log_reg = LogisticRegression(max_iter=1000)  # Ensure convergence with max_iter
          log_reg.fit(bijan_Robinson_X, bijan_Robinson_Y)
          log_reg_prediction = str(log_reg.predict(newInput))
          cleaned_value = log_reg_prediction.strip("[]").replace("'","")
          bijan_Robinson.append([amount,time,distance,cleaned_value])
          transaction_amounts = [row[0] for row in bijan_Robinson]
          times_of_day = [row[1] for row in bijan_Robinson]
          distances_from_home = [row[2] for row in bijan_Robinson]
          predictions = [row[3] for row in bijan_Robinson]

          labels = ["Average Amount", "Input Amount"]
          averages = [trans_avg_bijan, amount]  # Replace with actual `trans_avg_srikar` and `amount`

          # Create the scatter plot
          plt.figure(figsize=(8, 6))
          plt.style.use('ggplot')  # Apply a modern style

          # Plot the averages with customized colors and markers
          colors = ['blue', 'orange']
          markers = ['o', 's']
          for i, avg in enumerate(averages):
               plt.scatter(labels[i], avg, color=colors[i], s=150, label =f"{labels[i]} ({avg})", marker=markers[i])

          # Add labels and title
          plt.title("Comparison of Transaction Amounts", fontsize=16, weight='bold')
          plt.ylabel("Average Value", fontsize=14)
          plt.xlabel("Categories", fontsize=14)
          plt.ylim(0, max(averages) + 50)  # Adjust y-axis for better visualization

          # Add data labels for the dots
          for i, avg in enumerate(averages):
               plt.text(labels[i], avg + 5, f"{avg}", ha='center', fontsize=12, color=colors[i])

          # Add legend and grid
          plt.legend(fontsize=12, loc="lower center")
          plt.grid(axis='y', linestyle='--', alpha=0.7)

          output_path = "static/graph_avgAmount.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')
          plt.close()

          plt.figure(figsize=(8, 6))
          sns.boxplot(
          x=predictions,
          y=transaction_amounts,
          palette="Set2"
          )
          plt.title("Transaction Amount by Logistic Regression Predictions")
          plt.xlabel("Label")
          plt.ylabel("Transaction Amount")
          output_path2 = "static/graph_boxPlot.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path2, format="png", dpi=300, bbox_inches='tight')
          plt.close()

               # Heatmap: Mean Values of Indicators by Fraud/Legit
               # Calculate means manually
               #   # Unique labels ('fraud', 'legit')
          labels = list(set(predictions))
          heatmap_data = {
          label: [
                    np.mean([transaction_amounts[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([times_of_day[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([distances_from_home[i] for i in range(len(predictions)) if predictions[i] == label])
               ]
               for label in labels
               }

               # Convert to a 2D matrix for the heatmap
          heatmap_matrix = np.array(list(heatmap_data.values()))
          heatmap_labels = ["Transaction_Amount", "Time_of_Day", "Distance_From_Home"]

          plt.figure(figsize=(8, 4))
          sns.heatmap(
          heatmap_matrix,
               annot=True,
               cmap="coolwarm",
               fmt=".2f",
               xticklabels=heatmap_labels,
               yticklabels=labels,
               linewidths=0.5
               )
          plt.title("Mean Values of Indicators by Fraud/Legit")
          plt.xlabel("Indicators")
          plt.ylabel("Label")
          output_path3 = "static/graph_heatmap.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path3, format="png", dpi=300, bbox_inches='tight')
          plt.close()

          
         
     if name == "Tom Cruise":
          log_reg = LogisticRegression(max_iter=1000)  # Ensure convergence with max_iter
          log_reg.fit(tom_Cruise_X, tom_Cruise_Y)
          log_reg_prediction = str(log_reg.predict(newInput))
          cleaned_value = log_reg_prediction.strip("[]").replace("'","")
          tom_Cruise.append([amount,time,distance,cleaned_value])
          transaction_amounts = [row[0] for row in tom_Cruise]
          times_of_day = [row[1] for row in tom_Cruise]
          distances_from_home = [row[2] for row in tom_Cruise]
          predictions = [row[3] for row in tom_Cruise]

          labels = ["Average Amount", "Input Amount"]
          averages = [trans_avg_tom, amount]  # Replace with actual `trans_avg_srikar` and `amount`

          # Create the scatter plot
          plt.figure(figsize=(8, 6))
          plt.style.use('ggplot')  # Apply a modern style

          # Plot the averages with customized colors and markers
          colors = ['blue', 'orange']
          markers = ['o', 's']
          for i, avg in enumerate(averages):
               plt.scatter(labels[i], avg, color=colors[i], s=150, label=f"{labels[i]} ({avg})", marker=markers[i])

          # Add labels and title
          plt.title("Comparison of Transaction Amounts", fontsize=16, weight='bold')
          plt.ylabel("Average Value", fontsize=14)
          plt.xlabel("Categories", fontsize=14)
          plt.ylim(0, max(averages) + 50)  # Adjust y-axis for better visualization

          # Add data labels for the dots
          for i, avg in enumerate(averages):
               plt.text(labels[i], avg + 5, f"{avg}", ha='center', fontsize=12, color=colors[i])

          # Add legend and grid
          plt.legend(fontsize=12, loc="lower center")
          plt.grid(axis='y', linestyle='--', alpha=0.7)

          # Show the plot
          output_path = "static/graph_avgAmount.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')
          plt.close()

          plt.figure(figsize=(8, 6))
          sns.boxplot(
          x=predictions,
          y=transaction_amounts,
          palette="Set2"
          )
          plt.title("Transaction Amount by Logistic Regression Predictions")
          plt.xlabel("Label")
          plt.ylabel("Transaction Amount")
          output_path2 = "static/graph_boxPlot.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path2, format="png", dpi=300, bbox_inches='tight')
          plt.close()

               # Heatmap: Mean Values of Indicators by Fraud/Legit
               # Calculate means manually
          labels = list(set(predictions))  # Unique labels ('fraud', 'legit')
          heatmap_data = {
          label: [
                    np.mean([transaction_amounts[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([times_of_day[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([distances_from_home[i] for i in range(len(predictions)) if predictions[i] == label])
               ]
               for label in labels
               }

               # Convert to a 2D matrix for the heatmap
          heatmap_matrix = np.array(list(heatmap_data.values()))
          heatmap_labels = ["Transaction_Amount", "Time_of_Day", "Distance_From_Home"]

          plt.figure(figsize=(8, 4))
          sns.heatmap(
          heatmap_matrix,
               annot=True,
               cmap="coolwarm",
               fmt=".2f",
               xticklabels=heatmap_labels,
               yticklabels=labels,
               linewidths=0.5
               )
          plt.title("Mean Values of Indicators by Fraud/Legit")
          plt.xlabel("Indicators")
          plt.ylabel("Label")
          output_path3 = "static/graph_heatmap.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path3, format="png", dpi=300, bbox_inches='tight')
          plt.close()
         
     else:
          log_reg = LogisticRegression(max_iter=1000)  # Ensure convergence with max_iter
          log_reg.fit(sam_Sulek_X, sam_Sulek_Y)
          log_reg_prediction = str(log_reg.predict(newInput))
          cleaned_value = log_reg_prediction.strip("[]").replace("'","")
          sam_Sulek.append([amount,time,distance,cleaned_value])
          transaction_amounts = [row[0] for row in sam_Sulek]
          times_of_day = [row[1] for row in sam_Sulek]
          distances_from_home = [row[2] for row in sam_Sulek]
          predictions = [row[3] for row in sam_Sulek]


          labels = ["Average Amount", "Input Amount"]
          averages = [trans_avg_sam, amount]  # Replace with actual `trans_avg_srikar` and `amount`

          # Create the scatter plot
          plt.figure(figsize=(8, 6))
          plt.style.use('ggplot')  # Apply a modern style

          # Plot the averages with customized colors and markers
          colors = ['blue', 'orange']
          markers = ['o', 's']
          for i, avg in enumerate(averages):
               plt.scatter(labels[i], avg, color=colors[i], s=150, label=f"{labels[i]} ({avg})", marker=markers[i])

          # Add labels and title
          plt.title("Comparison of Transaction Amounts", fontsize=16, weight='bold')
          plt.ylabel("Average Value", fontsize=14)
          plt.xlabel("Categories", fontsize=14)
          plt.ylim(0, max(averages) + 50)  # Adjust y-axis for better visualization

          # Add data labels for the dots
          for i, avg in enumerate(averages):
               plt.text(labels[i], avg + 5, f"{avg}", ha='center', fontsize=12, color=colors[i])

          # Add legend and grid
          plt.legend(fontsize=12, loc="lower center")
          plt.grid(axis='y', linestyle='--', alpha=0.7)

          # Show the plot
          output_path = "static/graph_avgAmount.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')
          plt.close()


          plt.figure(figsize=(8, 6))
          sns.boxplot(
          x=predictions,
          y=transaction_amounts,
          palette="Set2"
          )
          plt.title("Transaction Amount by Logistic Regression Predictions")
          plt.xlabel("Label")
          plt.ylabel("Transaction Amount")
          output_path2 = "static/graph_boxPlot.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path2, format="png", dpi=300, bbox_inches='tight')
          plt.close()

               # Heatmap: Mean Values of Indicators by Fraud/Legit
               # Calculate means manually
          labels = list(set(predictions))  # Unique labels ('fraud', 'legit')
          heatmap_data = {
          label: [
                    np.mean([transaction_amounts[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([times_of_day[i] for i in range(len(predictions)) if predictions[i] == label]),
                    np.mean([distances_from_home[i] for i in range(len(predictions)) if predictions[i] == label])
               ]
               for label in labels
               }

               # Convert to a 2D matrix for the heatmap
          heatmap_matrix = np.array(list(heatmap_data.values()))
          heatmap_labels = ["Transaction_Amount", "Time_of_Day", "Distance_From_Home"]

          plt.figure(figsize=(8, 4))
          sns.heatmap(
          heatmap_matrix,
               annot=True,
               cmap="coolwarm",
               fmt=".2f",
               xticklabels=heatmap_labels,
               yticklabels=labels,
               linewidths=0.5
               )
          plt.title("Mean Values of Indicators by Fraud/Legit")
          plt.xlabel("Indicators")
          plt.ylabel("Label")
          output_path3 = "static/graph_heatmap.png"  # Save to a folder like 'static' in your project
          plt.savefig(output_path3, format="png", dpi=300, bbox_inches='tight')
          plt.close()

     return output_path,output_path2,output_path3,cleaned_value

print(creatingDataGraphs('Srikar Bond',333,22,33))