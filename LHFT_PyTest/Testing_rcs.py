import numpy as np
import matplotlib.pyplot as plt

# Define the data for each TX-RX pair (manually included based on the provided data)
tx3_rx1 = np.array([
    [-59.700618544525845, 84.669076218267],
    [-58.807958386772256, 85.31158667118018],
    [-56.89576269935389, 85.70403850033708],
    [-55.64691691181028, 86.56694446960776],
    [-55.24473597449057, 86.39409208784612],
    [-53.59370924962724, 87.08414567535516],
    [-53.64816607148817, 87.47610147753659],
    [-51.98125306926161, 88.29872318716895],
    [-51.94209202879651, 87.81736451808148],
    [-49.47145690117557, 88.42079234717227],
    [-48.83308461388093, 88.90417228867986],
    [-46.51107794851745, 89.46659213029403],
    [-46.34241876989848, 89.15347974352791],
    [-44.685969838991895, 89.94253627092462],
    [-43.705263779469426, 89.92950954389107],
    [-42.36138856971217, 90.63493785192213],
    [-41.39760553885217, 90.61932693301316],
    [-40.03079989129449, 91.63063015814743],
    [-39.41885355207928, 91.26609716611144],
    [-38.03634005749949, 92.3231509337707],
    [-37.1100143195272, 92.04224506566807],
    [-33.65029844650351, 92.94747538369936],
    [-32.561873930228224, 93.70765610920135],
    [-29.696337448762172, 93.98202431859225],
    [-28.2522730415131, 94.6159787547499],
    [-24.926901520035017, 94.62779070854594],
    [-19.338447684517682, 95.14376608486411],
    [-16.177094607158345, 95.91144554474327],
    [-14.569602251757942, 95.74636721960053],
    [-10.136158528484046, 95.83110339092309],
    [-7.1801357156564904, 95.91637100861632],
    [-6.926828803142925, 96.25472606691109],
    [-2.093012956465685, 95.78504470546554],
    [1.6826182591194794, 95.7836865647405],
    [5.455299631450217, 96.38023800782871],
    [6.281991349234076, 95.99785945211816],
    [10.217056476018485, 95.65112022005816],
    [12.705098192532688, 95.72770702914546],
    [13.820262869121493, 95.04550755588798],
    [15.293550307796025, 94.74281932299637],
    [16.65110779485174, 94.94638625733643],
    [18.56666945511985, 94.00782899236783],
    [18.620679790789595, 94.38241688558746],
    [21.18788105442897, 93.61839690186449],
    [21.913312132582575, 93.94795247471882],
    [23.969707556872166, 92.96991423046073],
    [24.377208080442088, 93.34047706457011],
    [26.91687293018863, 92.4077030198948],
    [30.84839508116859, 91.80197225653113],
    [31.45422211203342, 92.29806046504976],
    [33.95735634955393, 91.06704097549934],
    [34.41158381050917, 91.60375177004019],
    [36.736820868127495, 90.24589728322647],
    [38.699629463780454, 89.72720562545825],
    [40.492965716788305, 88.82008572033126],
    [40.9659532471, 89.17504207570289],
    [41.9615291877236, 88.17207544570144],
    [43.431273650593695, 87.61039568150616],
    [44.40677298874614, 86.91939730044926],
    [46.04303731442434, 86.53031950752636],
    [46.04507463967464, 87.26679571471482],
    [48.49241458721872, 85.57979814879513],
    [50.29047480796578, 85.01800028540639],
    [51.61508717894928, 85.14173432948255],
    [52.90341946373127, 84.02425462186115],
    [54.69734621270649, 83.16029997195145],
    [54.72549477689152, 83.84078458670456],
    [56.16177621186998, 82.21013291080067],
    [56.359796129512134, 83.01697093081467],
    [58.15486719334801, 81.97646144530636],
    [58.2863807025918, 81.5187211825666],
    [59.91614957262854, 80.65482558225362]
])
print(f'Length of tx3_rx1: {len(tx3_rx1)}')

tx3_rx2 = np.array([
    [-59.89430122183457, 82.51087250699983],
    [-57.748438876286166, 83.37340996658776],
    [-56.25979854245911, 84.19301836934538],
    [-53.95036881393963, 85.01233152411929],
    [-51.47323823067724, 86.09057716060013],
    [-49.99227434442645, 86.34903724553314],
    [-47.99956670879221, 87.47818499127399],
    [-47.35807183383443, 86.90924076980991],
    [-45.21279998425343, 87.72861297418056],
    [-43.726521634296, 88.37556035606906],
    [-41.414729921906904, 89.3675345317121],
    [-38.61577903640899, 89.9708442616094],
    [-35.80678971946519, 91.30796333020042],
    [-32.840728475191774, 92.12704028658737],
    [-31.186749270491433, 93.0329201501828],
    [-27.43898427107718, 94.00909931772996],
    [-27.236921744521915, 93.76531229855476],
    [-25.132189925979418, 94.50261780104711],
    [-24.277946451857353, 94.06640619233438],
    [-22.638492235662078, 94.70060693903433],
    [-19.51028201103243, 94.58267681663624],
    [-16.713693109404126, 95.01332552566443],
    [-13.873110169208687, 95.56399321067411],
    [-11.554536309066904, 95.95310406644386],
    [-9.808433266574482, 95.78782003651233],
    [-9.238537052841352, 96.21223318283025],
    [-7.955659866401888, 96.37870855148341],
    [-7.34016012282315, 96.218586844734],
    [-6.267444133478918, 96.21116043119848],
    [-4.883106402452526, 95.8292138038274],
    [-3.1407307107337203, 95.7334283711816],
    [-0.29672422362081363, 95.09375107642494],
    [0.15790904019719676, 95.60225468554074],
    [2.1437868609380075, 95.86150299655289],
    [4.140852971424934, 95.48064403426845],
    [5.122603592049245, 96.25037546307112],
    [8.09517963531323, 96.59685863874344],
    [8.100562121855049, 96.59592068312824],
    [8.254838376332941, 96.21297713304367],
    [10.578189290601273, 96.68168121635732],
    [12.221072656276064, 96.29114002507855],
    [13.995049675473254, 95.82242310020224],
    [16.341583190697094, 95.98635936702885],
    [18.484797917431507, 95.85560286257814],
    [19.568150615838093, 95.21610184087118],
    [22.927706075588475, 95.16075779175269],
    [24.978865165166653, 94.73933539678869],
    [26.55818366461493, 95.1161194044083],
    [30.392532194332233, 94.47839522879258],
    [32.8419094671266, 93.52787387006136],
    [33.95931172255307, 93.7702928850344],
    [35.78139839286682, 92.4045143416708],
    [36.255572348753944, 93.03289533281523],
    [37.579458613613895, 91.84271647828206],
    [39.86881147924163, 91.19441095566852],
    [40.34862044140155, 91.34164278800998],
    [42.9700963000507, 89.89833135681211],
    [42.973286100476315, 90.51747154824284],
    [44.93770888858164, 89.69353869772719],
    [46.72860313258113, 88.64518081478602],
    [47.23482771608796, 88.99946839196915],
    [47.645134500812404, 88.69982547993018],
    [49.34331927624879, 87.78093091689261],
    [51.49636933169957, 87.48146523569545],
    [51.79623952484758, 87.08940108946506],
    [53.755505144696116, 86.31171790039318],
    [54.446007218426544, 86.3972113225359],
    [55.38173103892845, 85.18883076877654],
    [57.501611561911034, 84.15209699880424],
    [57.872805030966774, 84.40290644175435],
    [59.61853960505664, 82.89953695274556],
    [59.82435479949082, 82.92906489432204]
      
    
])
print(f'Length of tx3_rx2: {len(tx3_rx2)}')


tx3_rx3 = np.array([
    [-60.37074296394124, 81.41267086549601],
    [-59.96273695612926, 81.45724258289702],
    [-59.913787588758915, 81.08641908482966],
    [-58.04444529205067, 82.19172693941576],
    [-57.60140538040241, 82.12155851568997],
    [-55.61674843395548, 83.19998130096103],
    [-53.79802085434924, 84.14896737017699],
    [-53.37125311693946, 84.78969300232191],
    [-51.98106476264521, 84.96845767374114],
    [-51.53927939697054, 85.61225511464139],
    [-50.49537690865519, 85.57223980041236],
    [-49.17502792553845, 86.08975046624577],
    [-48.710933961409545, 86.7377503683114],
    [-47.35452885803, 87.16823230111358],
    [-44.72115609251409, 88.16611916601904],
    [-44.223719238850684, 88.03041546312106],
    [-41.417682401743924, 89.1517082556257],
    [-38.77757493147786, 90.14356433207524],
    [-38.23830343137489, 90.46013893325578],
    [-36.469326194893185, 90.8765469764146],
    [-35.083269365550834, 91.41220600645559],
    [-34.821842445834285, 91.30760903261998],
    [-32.255782131295256, 92.49437401366447],
    [-32.02171056840157, 91.99724927295183],
    [-29.768714748189424, 93.05673425796579],
    [-29.5481229609436, 92.81650337812901],
    [-27.079849817192283, 93.24727018635069],
    [-24.44919028240468, 93.54848217932378],
    [-21.816168763747477, 94.02235519316599],
    [-19.34907661193096, 94.3667914909531],
    [-18.014217534959776, 94.82891995365713],
    [-14.90972792898301, 94.88318021444846],
    [-12.056584072737337, 95.6066648866936],
    [-11.948981148416223, 95.3137698738799],
    [-9.571233092242338, 96.08237063807267],
    [-8.126376602274775, 96.64048865619546],
    [-7.66906637666753, 96.17553968871022],
    [-6.592416361131114, 96.4712431045909],
    [-2.752596952056635, 95.5694546277661],
    [0.8164355085557986, 95.51536180336701],
    [1.0177197998218759, 95.17960919008557],
    [1.9761515392793996, 95.73158085448243],
    [4.618839425767973, 95.81728179039864],
    [7.595081552962938, 96.07617251753354],
    [8.902612452575795, 95.56526210639753],
    [12.511133309385457, 95.34813673918285],
    [13.536409190382429, 96.03069976780887],
    [16.119654166195104, 95.13101137196817],
    [20.703087199855574, 95.15706806282722],
    [21.203233949581488, 94.74069353751372],
    [25.137708580398495, 94.35078905023644],
    [29.894153597842745, 94.04691982541003],
    [32.34707384644153, 93.35538999798247],
    [32.642258785835864, 93.94407864938185],
    [34.937661210731335, 93.16335385070158],
    [35.29010574798616, 92.49102200089558],
    [36.24699033569975, 92.59962286820412],
    [39.21277045945507, 91.23781240927275],
    [41.49680886137615, 90.20101958970372],
    [44.431573819377135, 88.73233801957494],
    [44.59900544004273, 89.26038542774184],
    [48.19421412367937, 87.78134426406979],
    [48.650370102906635, 88.08900523560209],
    [50.80893026734708, 86.91709436617639],
    [52.15317939744732, 87.3079178605995],
    [54.24207382183755, 85.87988819943017],
    [55.425214907910245, 85.83359953466419],
    [55.483372449900685, 85.73298429319371],
    [56.52079776005198, 84.45460808290564],
    [58.80660764987525, 83.54731102898843],
    [59.78210698802769, 82.85631264793153],
    [60.332767647589804, 83.24607329842931]
]) 
print(f'Length of tx3_rx3: {len(tx3_rx3)}')


tx3_rx4 = np.array([
    [-60.06082108464267, 82.33827053572746],
    [-59.518835134761446, 82.75551751922609],
    [-58.57513323065264, 82.94205266239868],
    [-57.088854880695195, 83.58900004428719],
    [-55.2736702768934, 84.2789945821995],
    [-54.845642959650235, 85.35348358213226],
    [-54.115707684812925, 84.92606006328148],
    [-52.95774509273246, 85.57312554436346],
    [-50.97840260999217, 86.26306103267903],
    [-48.50540549850161, 87.03914988263892],
    [-46.541973195512554, 87.90681125769402],
    [-46.526063015761316, 87.72908537095448],
    [-43.72593113832859, 88.41872561128635],
    [-40.925799260895865, 89.1083658516182],
    [-38.245169041818244, 90.11352096156689],
    [-37.957376032752826, 90.10010382887427],
    [-34.992495780414224, 90.83285027482667],
    [-32.763837304103625, 91.84464410868642],
    [-32.0258440401734, 91.6950924864309],
    [-28.566128167149714, 92.60032280446218],
    [-26.099626511300613, 92.90159384703202],
    [-25.469412275139334, 93.445129945981],
    [-24.12855097210398, 92.9872157623057],
    [-20.509991683848455, 93.50389973378473],
    [-16.89084189962552, 94.06374896048106],
    [-14.552233468897356, 94.61103217778117],
    [-13.766527736087667, 94.4511143150985],
    [-11.462412471274817, 94.8819401729169],
    [-11.238146094468888, 95.25974892844032],
    [-8.089977639088204, 95.86519802995123],
    [-7.183678691460926, 95.65737947731265],
    [-5.941613704521288, 95.99440500426716],
    [-2.9132118551906956, 95.82850520866651],
    [-1.8150957609623362, 95.99291507144525],
    [0.9960856484902791, 96.25186539589303],
    [1.8485476259601796, 95.91312328079559],
    [5.126036397270923, 96.42368444891557],
    [6.281400853266675, 95.95469419690087],
    [9.890512206043724, 95.78073408490349],
    [10.73209339137317, 96.11836741504999],
    [15.137068876433815, 95.30402669041773],
    [15.674474709284311, 95.63997978459147],
    [17.59412259680444, 94.91465364951112],
    [18.312013387940354, 95.465717241741],
    [20.613423221973775, 94.9882831682885],
    [21.856322489530996, 94.48146580782308],
    [23.24924549801898, 94.72736613251581],
    [24.565440233430763, 94.51025312170725],
    [26.60981502713821, 93.96177030691028],
    [30.496469455185206, 93.94485341444926],
    [30.706676048991483, 93.44231100438444],
    [33.65206993440576, 92.75060402816666],
    [33.77966158261856, 93.03378929250837],
    [36.26973855791043, 92.10218040635964],
    [38.06366530688568, 91.23822575644994],
    [39.52674965791144, 91.5152497604188],
    [39.85936354376315, 90.50376687219207],
    [41.16276741314289, 90.77809059745114],
    [41.80498805669852, 89.8679800325163],
    [41.97806307481093, 89.38070259178522],
    [44.096762605858714, 88.25763831137837],
    [44.09781587767772, 88.95727349445268],
    [47.69465453525507, 87.26353835025267],
    [50.14698428788648, 86.52884326760785],
    [51.47920530559118, 86.61492030647322],
    [52.765833903325984, 85.96675015623539],
    [54.23085439845687, 85.05974835030189],
    [56.18598654653357, 83.97990837470905],
    [56.88329892581804, 84.44659127209272],
    [58.182329635121434, 83.36293333206193],
    [58.963089081237484, 82.98610366156709],
    [59.2825473996034, 82.33850673411442]
    
])
print(f'Length of tx3_rx4: {len(tx3_rx4)}')
# # Combine all data into one array for sorting
# # Round the data to 4 decimal places
# tx3_rx1 = np.round(tx3_rx1, 4)
# tx3_rx2 = np.round(tx3_rx2, 4)
# tx3_rx3 = np.round(tx3_rx3, 4)
# tx3_rx4 = np.round(tx3_rx4, 4)
# combined_data = np.hstack((tx3_rx1, tx3_rx2, tx3_rx3, tx3_rx4))

# # Sort data by the first column (angle)
# sorted_indices = np.argsort(combined_data[:, 0])
# sorted_data = combined_data[sorted_indices]

# # Split sorted data back into separate arrays
# sorted_tx3_rx1 = sorted_data[:, :2]
# sorted_tx3_rx2 = sorted_data[:, 2:4]
# sorted_tx3_rx3 = sorted_data[:, 4:6]
# sorted_tx3_rx4 = sorted_data[:, 6:8]

# # Save sorted data to a text file in the specified format
# Define the function to read data from the file
def read_data(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    angles_rx1, values_rx1 = [], []
    angles_rx2, values_rx2 = [], []
    angles_rx3, values_rx3 = [], []
    angles_rx4, values_rx4 = [], []

    for line in lines[1:]:  # Skip the header
        parts = line.split('||')
        angle_rx1, value_rx1 = map(float, parts[0].split())
        angle_rx2, value_rx2 = map(float, parts[1].split())
        angle_rx3, value_rx3 = map(float, parts[2].split())
        angle_rx4, value_rx4 = map(float, parts[3].split())
        
        angles_rx1.append(angle_rx1)
        values_rx1.append(value_rx1)
        angles_rx2.append(angle_rx2)
        values_rx2.append(value_rx2)
        angles_rx3.append(angle_rx3)
        values_rx3.append(value_rx3)
        angles_rx4.append(angle_rx4)
        values_rx4.append(value_rx4)

    return (np.array(angles_rx1), np.array(values_rx1), 
            np.array(angles_rx2), np.array(values_rx2), 
            np.array(angles_rx3), np.array(values_rx3), 
            np.array(angles_rx4), np.array(values_rx4))

# Function to process and plot the data
def process_and_plot(file_path, plot_title):
    # Read the data from the file
    angles_rx1, values_rx1, angles_rx2, values_rx2, angles_rx3, values_rx3, angles_rx4, values_rx4 = read_data(file_path)

    # Define a common set of angles for interpolation
    common_angles = np.linspace(min(angles_rx1.min(), angles_rx2.min(), angles_rx3.min(), angles_rx4.min()),
                                max(angles_rx1.max(), angles_rx2.max(), angles_rx3.max(), angles_rx4.max()), 500)

    # Interpolate the values to the common set of angles
    interp_values_rx1 = np.interp(common_angles, angles_rx1, values_rx1)
    interp_values_rx2 = np.interp(common_angles, angles_rx2, values_rx2)
    interp_values_rx3 = np.interp(common_angles, angles_rx3, values_rx3)
    interp_values_rx4 = np.interp(common_angles, angles_rx4, values_rx4)

    # Calculate the average of the interpolated values
    average_values = np.mean([interp_values_rx1, interp_values_rx2, interp_values_rx3, interp_values_rx4], axis=0)

    # Plot the averaged data
    plt.plot(common_angles, average_values, label='Average Pattern')
    plt.xlabel('Angle [degrees]')
    plt.ylabel('dBFS')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)

# File paths
azimuth_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/LHFT_PyTest/azimuth_radiation_pattern.txt'
elevation_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/LHFT_PyTest/elevation_radiation_pattern.txt'

# Process and plot for azimuth
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
process_and_plot(azimuth_file, 'Averaged Azimuth Antenna Gain Pattern')

# Process and plot for elevation
plt.subplot(1, 2, 2)
process_and_plot(elevation_file, 'Averaged Elevation Antenna Gain Pattern')

# Show the plots
plt.tight_layout()
plt.show()


# import numpy as np
# from scipy.integrate import dblquad
# import matplotlib.pyplot as plt

# # Example RCS data
# azimuth_angles = np.rad2deg(2.5)  # Azimuth angles in radians
# elevation_angles = np.rad2deg(2.5)  # Elevation angles in radians

# # Create a meshgrid for azimuth and elevation
# phi, theta = np.meshgrid(azimuth_angles, elevation_angles)

# # Example RCS values (replace with actual data)
# # Assuming RCS values are given in dBsm, we need to convert them to linear scale
# rcs_dbsm = np.random.uniform(-40, 30, phi.shape)  # Random example values
# rcs_linear = 10 ** (rcs_dbsm / 10)  # Convert dBsm to linear scale

# # Define the integrand function
# def integrand(phi, theta):
#     # Interpolate RCS values at the given phi, theta
#     rcs_value = np.interp(phi, azimuth_angles, np.mean(rcs_linear, axis=0))
#     return rcs_value * np.cos(theta)

# # Perform the double integral over the angular extent
# effective_area, error = dblquad(integrand, 
#                                 0, np.pi,  # Azimuth angle range
#                                 lambda x: 0, lambda x: np.pi/2)  # Elevation angle range

# print(f"Effective Area: {effective_area:.4f} square meters")

# # Calculate the gain factor
# gain_factor = (4 * np.pi) / effective_area

# print(f"Gain Factor: {gain_factor:.4f}")



# import numpy as np
# from scipy.integrate import dblquad

# # Define the azimuth and elevation angles in radians
# azimuth_angle = np.radians(2.86)  # Example value, convert degrees to radians
# elevation_angle = np.radians(2.86)  # Example value, convert degrees to radians

# # Define the integrand function
# def integrand(phi, theta):
#     return np.cos(phi)

# # Perform the double integral
# effective_area, error = dblquad(integrand, 
#                                 -elevation_angle/2, elevation_angle/2, 
#                                 lambda x: -azimuth_angle/2, lambda x: azimuth_angle/2)

# print(f"Effective Area: {effective_area:.4f} square meters")

# # Calculate the gain factor
# gain_factor = (4 * np.pi) / effective_area

# print(f"Gain Factor: {gain_factor:.4f}")




# import math

# # Given values
# RCS_linear = 4.701194832873645e-05
# desired_RCS_dBsm = -20.5327

# # Calculate the constant k
# k = 10**(desired_RCS_dBsm / 10) / RCS_linear

# # Calculate the new RCS in linear
# RCS_new_linear = k * RCS_linear

# # Convert the new RCS in linear to dBsm
# RCS_new_dBsm = 10 * math.log10(RCS_new_linear)

# # Print the results
# print(f"Constant k: {k}")
# print(f"New RCS in linear: {RCS_new_linear}")
# print(f"New RCS in dBsm: {RCS_new_dBsm}")








# import numpy as np

# def calculate_azimuth_angle(width, distance):
#     """
#     Calculate the azimuth angle to ensure the rays from the Tx antenna exactly cover the plate width.

#     Parameters:
#     - width: float, the width of the plate in meters
#     - distance: float, the distance from the Tx antenna to the plate in meters

#     Returns:
#     - theta_degrees: float, the azimuth angle in degrees
#     """
#     # Calculate the half-angle theta/2 in radians
#     theta_half_radians = np.arctan((width / 2) / distance)
    
#     # Calculate the full theta in radians
#     theta_radians = 2 * theta_half_radians
    
#     # Convert theta from radians to degrees
#     theta_degrees = np.degrees(theta_radians)
    
#     return theta_degrees

# # Constants
# plate_width = 0.1  # in meters
# antenna_distance = 0.8  # in meters

# # Calculate the azimuth angle
# theta = calculate_azimuth_angle(plate_width, antenna_distance)
# print(f"The azimuth angle should be set to {theta:.2f} degrees.")



# To continue from where your code snippet leaves off and compute the Radar Cross Section (RCS) in Python, we need to focus on translating the radar signal processing
#  results (particularly, the FFT range and angle data) into RCS values. I'll outline how to integrate RCS calculation into your existing pipeline, considering you've
# already done substantial preprocessing, FFT, and normalization of your radar data.

# import matplotlib.pyplot as plt
# import numpy as np
# # Constants and radar specifications taken from IWR6843AOP user guide

# c = 299792458  # Speed of light in meters/second
# transmitted_power = 10  # Transmitted power in dBm(IWR6843AOP max transmit power)
# Gt = 6  # Gain of the transmitting antenna in dBi
# Gr = 6  # Gain of the receiving antenna in dBi
# carrier_frequency = 60e9  # Carrier frequency in Hz (example for automotive radar)
# wavelength = 0.00495
# bandwidth = 4e9

# # Converting gains from dB to linear scale
# Gt_linear = 10**(Gt / 10)
# Gr_linear = 10**(Gr / 10)

# range_resolution = c / (2 * bandwidth)  # Bandwidth of the chirp
# # accessing the first row of this array, to find the length of array 
# range_bins = np.arange(len(fft_range_angle_abs_norm_log[0])) * range_resolution 


# # The maximum value within each row is assumed to be the return from a target, to find the range
# # bin that has the maximum response for each row.
# target_bin = np.argmax(fft_range_angle_abs_norm_log, axis=1)
# target_range = range_bins[target_bin]

# # Calculating received power Pr from the normalized FFT log data
# Pr = 10**((fft_range_angle_abs_norm_log[target_bin] - 30) / 10)  # Convert dBm to watts
# #Converting transmitted_power inot watts
# Pt = 10**((transmitted_power - 30) / 10)

# # RCS calculation
# RCS = (Pr * (4 * np.pi)**3 * target_range**2) / (Pt * Gt_linear * Gr_linear * wavelength**2)
# RCS_dBsm = 10 * np.log10(RCS)  # Convert RCS to dBsm



# # Compute angles assuming uniform spacing across the FFT indices
# angles = np.linspace(-np.pi / 2, np.pi / 2, angular_dim)

# plt.figure(figsize=(10, 5))
# plt.plot(angles, RCS_dBsm)
# plt.xlabel('Angle (radians)')
# plt.ylabel('RCS (dBsm)')
# plt.title('RCS vs. Angle')
# plt.grid(True)
# plt.show()
# import numpy as np

# gt = 6
# Antenna_gain_dBsm = 6
# gt_linear = 10**(gt / 10)
# print(gt_linear)
# range_antenna_plate = 0.8

# Pt = (1200*600)*60
# Power_ratio = (162232/Pt)**2
# print(f" power ratio {Power_ratio}")
# RCS_without_const = Power_ratio*(4*np.pi*(range_antenna_plate**2))**2
# print(f" RCS without const factor {RCS_without_const}") 
# # Converting gains from dB to linear scale
# Antenna_gain_linear = 10**((-6) / 10)
# print(f" Antenna_gain_linear  {Antenna_gain_linear}") 
# RCS_final = (RCS_without_const / Antenna_gain_linear)
# print(f" RCS with const factor linear {RCS_final}") 
# RCS_dBsm = 10 * np.log10(RCS_final) 

# print(f" RCS without const factor dBsm {RCS_dBsm}")
# pi = np.pi
# print(pi)

# import numpy as np

# # Define the radius of the circle
# radius = 80  # in cm

# # Define the increment in degrees
# degree_increment = 5

# # Calculate the number of points around the circle
# num_points = int(360 / degree_increment)

# # List to store coordinates
# coordinates = []

# for i in range(num_points):
#     # Convert degrees to radians for calculation
#     angle_rad = np.deg2rad(i * degree_increment)
    
#     # Calculate x and y coordinates
#     x = radius * np.cos(angle_rad)
#     y = radius * np.sin(angle_rad)
    
#     # Append the coordinates to the list
#     coordinates.append((x, y, 0))  # z is always 0

# # Print out all coordinates
# for coord in coordinates:
#     print(f"x: {coord[0]:.2f}, y: {coord[1]:.2f}, z: {coord[2]}")
