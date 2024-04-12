# %%
# Import the different libraries

import cv2
import datetime
import os

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from cartes.crs import PlateCarree
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from traffic.core import Traffic

# %%
# Set variables

# We consider the following record from GOES 16.
record_id = 4258404969592519689

# GOES 16 file with contrail metadata
ct_metadata_filename = "data/metadata_contrail.csv"

# GRIB file
nc_filename = 'data/ERA5_2019_04_28_19_00.nc'
# Below the following altitude, we consider that no contrail can be produced.
filtre_altitude = 34000
# In ERA5, it corresponds to the fifth pressure level (250 hPa).
selected_level = 5

# We consider the last 40 minutes before the satellite photo.
delta_before = 40
delta_after = 0 # We do not need to look after.

# %%
# Read metadata from GOES 16.

ct_metadata = pd.read_csv(ct_metadata_filename)
west = ct_metadata.loc[ct_metadata['record_id'] == record_id, 'ouest'].values[0]
north = ct_metadata.loc[ct_metadata['record_id'] == record_id, 'nord'].values[0]
east = ct_metadata.loc[ct_metadata['record_id'] == record_id, 'est'].values[0]
south = ct_metadata.loc[ct_metadata['record_id'] == record_id, 'sud'].values[0]
dt_str = ct_metadata.loc[ct_metadata['record_id'] == record_id, 'timestamp'].values[0]
loc = ct_metadata.loc[ct_metadata['record_id'] == record_id, 'central_meridian'].values[0]
dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S%z")
dt_minus = dt - datetime.timedelta(minutes=delta_before)
dt_max = dt + datetime.timedelta(minutes=delta_after)

print('west:', west)
print('north:', north)
print('east:', east)
print('south:', south) 
print('loc:', loc)
print('dt:', dt)
print('dt_minus:', dt_minus)
print('dt_max:', dt_max)

# %%
# Read GOES 16 data

with open(os.path.join("data", str(record_id), 'human_pixel_masks.npy'), 'rb') as f:
    mask = np.load(f)

mask_gray = np.uint8(mask)

# %%
# Read GRIB data

nc_file = netCDF4.Dataset(nc_filename)
u_wind = nc_file.variables['u'][:]
v_wind = nc_file.variables['v'][:]
pressure = nc_file.variables['level'][:]  
latitude = nc_file.variables['latitude'][:]
longitude = nc_file.variables['longitude'][:]
nc_file.close()

wind_uv = np.sqrt(u_wind ** 2 + v_wind ** 2)
uv_wind = np.sqrt(u_wind**2 + v_wind**2)
theta_wind = np.arctan2(u_wind, v_wind)

# %%
# Read and plot the trajectories from Opensky

t = Traffic.from_file("data/article_traffic_40min.parquet")

print("Total number of flights:", len(t))

# Prepare plotting
projection = PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(5, 5))
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# Plot the trajectories in blue
for flight in t:
    flight.plot(ax=ax, color='blue', label=flight.callsign, zorder=0)

# Plot the contrails in red above the trajectories
colors = [(1, 1, 1, 0), (1, 0, 0, 1)]  # From transparent (no contrail) to red (contrail)
cmap = LinearSegmentedColormap.from_list('custom_reds', colors, N=256)
im = ax.imshow(mask, extent=[west, east, south, north], cmap=cmap, zorder=1)

# Set title, labels and ticks
ax.set_title('Trajectories between {0} et {1}'.format(dt_minus,dt))
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xticks(np.arange(round(west), round(east), 2), crs=PlateCarree())
ax.set_yticks(np.arange(round(south), round(north), 2), crs=PlateCarree())

plt.show()

# %%
# Transpose the constrails in the Hough space at the current date

def get_contour_line_equation(contour):
    contour = np.squeeze(contour)   
    x = contour[:, 0]
    y = contour[:, 1]
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)  
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
all_contour_points = []
for contour in contours:
    contour_points = contour.squeeze().tolist()
    all_contour_points.append(contour_points)

traj_contrail = []
equation_traj_contrail = []
mask_empty = np.zeros((256,256,3))
mask_empty = np.uint8(mask_empty)

dico_contrail = {}
compteur = 1
for contour_points in all_contour_points:
    contour_points = np.array(contour_points)
    if len(contour_points) > 1:
        
        if len(contour_points.shape) == 2: 
            s, i = get_contour_line_equation(contour_points)
            contour = np.squeeze(contour_points)

            x_min = min(contour[:, 0])
            x_max = max(contour[:,0])
            
            y_min = int(s * x_min + i)
            y_max = int(s * x_max + i)
            equation_traj_contrail.append([s, i, x_min, x_max])

            thickness = 1  
            color = (255,255,255)
            image_with_line = cv2.line(mask_empty.copy(), (x_min, y_min), (x_max, y_max), color, thickness)

            traj_contrail.append(image_with_line)

            gray_image_with_line = cv2.cvtColor(image_with_line, cv2.COLOR_BGR2GRAY)

            lines = cv2.HoughLinesP(gray_image_with_line, 1, np.pi/180, 5, 10, 1)

            if lines is None:
                pass
            elif len(lines) > 1:
                avg_x1 = np.mean(lines[:, 0, 0])
                avg_y1 = np.mean(lines[:, 0, 1])
                avg_x2 = np.mean(lines[:, 0, 2])
                avg_y2 = np.mean(lines[:, 0, 3])
                theta = np.arctan2(avg_x2 - avg_x1, avg_y2 - avg_y1)

                if avg_x1 == avg_x2:
                    rho = avg_x1
                elif avg_y1 == avg_y2:
                    rho = (256 - avg_y1)
                else:
                    a = (avg_y2 - avg_y1) / (avg_x2 - avg_x1)
                    b = avg_y1 - a * avg_x1
                    x_int = - b * a / (a**2 + 1)
                    y_int = b / (a**2 + 1)
                    rho = x_int * np.cos(theta) + (256 - y_int) * np.sin(theta)

                dico_contrail[str(compteur)] = (rho, np.rad2deg(theta),(avg_x1, avg_y1, avg_x2, avg_y2))
                compteur += 1

            elif len(lines) == 1:
                x1, y1, x2, y2 = lines[0][0]
                theta = np.arctan2(x2 - x1, y2 - y1)

                if x1 == x2:
                    rho = x1
                elif y1 == y2:
                    rho = (256 - y1)
                else:
                    a = (y2 - y1) / (x2 - x1)
                    b = y1 - a * x1
                    x_int = - b *a / (a**2 + 1)
                    y_int = b / (a**2 + 1)
                    rho = x_int * np.cos(theta) + (256 - y_int) * np.sin(theta)

                dico_contrail[str(compteur)] = (rho, np.rad2deg(theta), (x1, y1, x2,y2))
                compteur += 1

rhos_a = []
thetas_a = []
for k, v in dico_contrail.items():
    rho, theta, *_ = v
    rhos_a.append(rho)
    thetas_a.append(theta)

# %%
# Determine how the contrails can be advected in the Hough space (taking into account the GRIB prediction)

dico_contrail_before = {}
compteur = 1
for contour_points in all_contour_points:
    
    contour_points = np.array(contour_points)

    if len(contour_points) > 1:
        
        if len(contour_points.shape) == 2: 
            s, i = get_contour_line_equation(contour_points)
            contour = np.squeeze(contour_points)

            x_min = min(contour[:, 0])
            x_max = max(contour[:,0])
               
            y_min = int(s * x_min + i)
            y_max = int(s * x_max + i)

            x_p = int(x_min / (256/18))
            y_p = int(y_min / (256/18))
            
            equation_traj_contrail.append([s, i, x_min, x_max])

            thickness = 1  
            color = (255,255,255)
            image_with_line = cv2.line(mask_empty.copy(), (x_min, y_min), (x_max, y_max), color, thickness)

            traj_contrail.append(image_with_line)

            gray_image_with_line = cv2.cvtColor(image_with_line, cv2.COLOR_BGR2GRAY)
            
            lines = cv2.HoughLinesP(gray_image_with_line, 1, np.pi/180, 5, 10, 1)
            
            if lines is None:
                pass
            elif len(lines) > 1:
                avg_x1 = np.mean(lines[:, 0, 0])
                avg_y1 = np.mean(lines[:, 0, 1])
                avg_x2 = np.mean(lines[:, 0, 2])
                avg_y2 = np.mean(lines[:, 0, 3])
                theta = np.arctan2(avg_x2 - avg_x1, avg_y2 - avg_y1)

                if avg_x1 == avg_x2:
                    rho = avg_x1 - 600 * uv_wind[0,selected_level, x_p, y_p] * np.cos(theta_wind[0,selected_level, x_p, y_p] - theta) / 1950
                elif avg_y1 == avg_y2:
                    rho = 256 - avg_y1 - 600 * uv_wind[0,selected_level, x_p, y_p] * np.sin(theta_wind[0,selected_level, x_p, y_p] - theta) / 1950
                else:
                    a = (avg_y2 - avg_y1) / (avg_x2 - avg_x1)
                    b = avg_y1 - a * avg_x1
                    x_int = - b *a / (a**2 + 1) - 600 * uv_wind[0,selected_level, x_p, y_p] * np.cos(theta_wind[0,selected_level, x_p, y_p]- theta) / 1950
                    y_int = b / (a**2 + 1) - 600 * uv_wind[0,selected_level, x_p, y_p] * np.sin(theta_wind[0,selected_level, x_p, y_p]-theta) / 1950
                    rho = x_int * np.cos(theta) + (256 - y_int) * np.sin(theta)

                dico_contrail_before[str(compteur)] = (rho, np.rad2deg(theta),(avg_x1, avg_y1, avg_x2, avg_y2))
                compteur += 1

            elif len(lines) == 1:
                x1, y1, x2, y2 = lines[0][0]
                theta = np.arctan2(x2 - x1, y2 - y1)

                if x1 == x2:
                    rho = x1 - 600 * uv_wind[0,selected_level, x_p, y_p] * np.cos(theta_wind[0,selected_level, x_p, y_p]) / 1950
                elif y1 == y2:
                    rho = (256 - y1) - 600 * uv_wind[0,selected_level, x_p, y_p] * np.sin(theta_wind[0,selected_level, x_p, y_p]) / 1950
                else:
                    a = (y2 - y1) / (x2 - x1)
                    b = y1 - a * x1
                    x_int = - b *a / (a**2 + 1) - 600 * uv_wind[0,selected_level, x_p, y_p] * np.cos(theta_wind[0,selected_level, x_p, y_p]) / 1950
                    y_int = b / (a**2 + 1) - 600 * uv_wind[0,selected_level, x_p, y_p] * np.sin(theta_wind[0,selected_level, x_p, y_p]) / 1950
                    rho = x_int * np.cos(theta) + (256 - y_int) * np.sin(theta)

                dico_contrail_before[str(compteur)] = (rho, np.rad2deg(theta), (x1, y1, x2,y2))
                compteur += 1

rhos = []
thetas = []
for k, v in dico_contrail_before.items():
    rho, theta, *_ = v
    rhos.append(rho)
    thetas.append(theta)

# %%
# Plot the contrails in the Hough space

plt.figure(figsize=(8, 6))
plt.scatter(rhos, thetas, label="Contrail at $t - \Delta t$")
plt.scatter(rhos_a, thetas_a, label="Contrail at $t$")
plt.xlabel('Rho ')
plt.ylabel('Theta')
plt.legend()
plt.title('Wind impact on rho values')
plt.grid(True)

for rho, theta, rho_a, theta_a in zip(rhos, thetas, rhos_a, thetas_a):
    plt.arrow(rho, theta, rho_a - rho, theta_a - theta, head_width=0.5, head_length=0.3, fc='black', ec='black', alpha=0.7)

plt.show()

# %%
# Filter trajectories according to their altitude

fq = t.query("(altitude > {0}) and (-1 < vertical_rate < 1)".format(filtre_altitude))

print("Number of flights after filtering altitude:", len(fq))

# %%
# Keep trajectories in the Hough space

dico_traj = {}

cpt = 1
k = 0

for f in fq:

    position = f.coords
    xf = []
    yf = []

    for coords in list(position):
        lon, lat, _ = coords
        x = 0 + 256 * (lon - west) / (east - west)
        y = 256 - (0 + 256 * (lat - south) / (north - south))
        xf.append(x)
        yf.append(y)
    
    xf = np.array(xf)
    yf = np.array(yf)

    try :
        slope, intercept, r_value, p_value, std_err = linregress(xf, yf)
    except:
        slope = 0
        intercept = yf

    num_points = 1000
    y_values =  slope * xf + intercept
    x_min = xf.min()
    x_max = xf.max()

    if slope > 0:
        y_min = slope * x_min + intercept
        y_max = slope * x_max + intercept
    else:
        y_max = slope * x_min + intercept
        y_min = slope * x_max + intercept
    y_min = y_min.max()
    y_max = y_max.max()

    img = np.zeros((256,256,3))
    img = np.uint8(img)

    color = (0, 255, 0)
    thickness = 2

    if slope > 0:
        img = cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
    else:
        img = cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_min)), color, thickness)

    gray_img_with_line = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLinesP(gray_img_with_line, 1, np.pi/180, 5, 10, 1)

    if lines is None:
        k += 1
        # print("No lines " , k, " for ", f.callsign)
        continue
    elif len(lines) > 1:
        avg_x1 = np.mean(lines[:, 0, 0])
        avg_y1 = np.mean(lines[:, 0, 1])
        avg_x2 = np.mean(lines[:, 0, 2])
        avg_y2 = np.mean(lines[:, 0, 3])
        theta = np.arctan2(avg_x2 - avg_x1, avg_y2 - avg_y1)
        if avg_x1 == avg_x2:
            rho = avg_x1
        elif avg_y1 == avg_y2:
            rho = 256 - avg_y1
        else:
            a = (avg_y2 - avg_y1) / (avg_x2 - avg_x1)
            b = avg_y1 - a * avg_x1

            x_int = - b *a / (a**2 + 1)
            y_int = b / (a**2 + 1)
            rho = x_int * np.cos(theta) + (256 - y_int) * np.sin(theta)

    elif len(lines) == 1:
        x1, y1, x2, y2 = lines[0][0]
        theta = np.arctan2(x2 - x1, y2 - y1)

        if x1 == x2:
            rho = x1
        elif y1 == y2:
            rho = 256 - y1
        else:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            x_int = - b *a / (a**2 + 1)
            y_int = b / (a**2 + 1)

            rho = x_int * np.cos(theta) + (256 - y_int) * np.sin(theta)
    
    cpt=cpt+1
    dico_traj[f.callsign] = (rho, np.rad2deg(theta), [])  

print('Number of flights in the Hough space:', len(dico_traj))

# %%
# Trajectories in the Hough space after the first filtering (figure 2)

rhos_contrail = []
thetas_contrail = []

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] #plt.cm.tab20.colors
color_contrail = []

cc = 0
for k, v in dico_contrail.items():
    rho, theta, *_ = v
    rhos_contrail.append(rho)
    thetas_contrail.append(theta)
    color_contrail.append(color_cycle[cc % len(color_cycle)])
    cc += 1

rhos_traj = []
thetas_traj = []
names = []
for k, v in dico_traj.items():
    rho, theta, *_ = v
    rhos_traj.append(rho)
    thetas_traj.append(theta)
    final_name = k
    # final_name = k + ' FL' + str(int(fq[k].mean('altitude') /100))
    names.append(final_name)

# Plot the contrail and aicraft position in the Hough space
plt.figure(figsize=(8, 6))
plt.scatter(rhos_traj, thetas_traj, color='blue', s=35, label='Aircraft position')
plt.scatter(rhos_contrail, thetas_contrail, color='red',s=40, label='Contrail')

# Plot the callsign
# for i, name in enumerate(names):
#     plt.text(rhos_traj[i], thetas_traj[i], name, fontsize=10, ha='left', va='bottom', color='blue')

plt.xlabel('Rho',fontsize=15)
plt.ylabel('Theta',fontsize=15)
plt.title("Aircraft and contrails in the Hough Space",fontsize=20)
plt.grid(True)
plt.legend()
plt.show()

# %%
# Determine and plot the suspects

theta_lim = 3
rho_max = 70

suspect = []
for k_vol, v_vol in dico_traj.items():
    for k_c, v_c in dico_contrail.items():
        
        rho_c, theta_c, *_ = v_c
        rho_v, theta_v, *_ = v_vol
        rho_cb,  theta_cb, *_ = dico_contrail_before[k_c]

        if rho_cb < rho_c:
            rho_lim_gauche = rho_c - rho_max
            rho_lim_droite = rho_c
            if  theta_c - theta_lim < theta_v < theta_c + theta_lim:
                if rho_lim_gauche < rho_v < rho_lim_droite:
                    suspect.append(k_vol)
                    dico_traj[k_vol][2].append(k_c)
        else: 
            rho_lim_gauche = rho_c
            rho_lim_droite =  rho_c + rho_max
            if  theta_c - theta_lim < theta_v < theta_c + theta_lim:
                if rho_lim_gauche < rho_v < rho_lim_droite:
                    suspect.append(k_vol)
                    dico_traj[k_vol][2].append(k_c)

rhos_traj = []
thetas_traj = []
names = []

for k, v in dico_traj.items():
    if k in suspect:
        rho, theta, *_ = v
        rhos_traj.append(rho)
        thetas_traj.append(theta)
        # final_name = k + ' FL' + str(int(fq[k].mean('altitude') /100))
        final_name = k
        names.append(final_name)

plt.figure(figsize=(16, 16))
# Determine the suspect domains (according to contrail direction)

for k_vol, v_vol in dico_traj.items():
    cc = 0
    for k_c, v_c in dico_contrail.items():
        if k_vol in suspect:
            
            rho_c, theta_c, *_ = v_c            
            rho_cb,  theta_cb, *_ = dico_contrail_before[k_c]

            if rho_cb < rho_c:
                rho_lim_gauche =  rho_c - rho_max
                rho_lim_droite = rho_c
                domain_points = [(theta_c - theta_lim, rho_lim_gauche), (theta_c + theta_lim, rho_lim_gauche), 
                (theta_c + theta_lim, rho_lim_droite ), (theta_c - theta_lim, rho_lim_droite)]
            else: 
                rho_lim_gauche = rho_c
                rho_lim_droite = rho_c + rho_max
                domain_points = [(theta_c- theta_lim, rho_lim_gauche), (theta_c + theta_lim, rho_lim_gauche), 
                (theta_c + theta_lim, rho_lim_droite), (theta_c - theta_lim, rho_lim_droite)]
            
            domain_points.append(domain_points[0])  # To close the polygon
            domain_points = np.array(domain_points)
            plt.plot(domain_points[:, 1], domain_points[:, 0], color=color_contrail[cc], alpha=1)
        cc += 1

plt.scatter(rhos_traj, thetas_traj, color='blue', s=75, label='Aircraft positions')
plt.scatter(rhos_contrail, thetas_contrail, color='red',s=100, label='Contrail')

# for i, name in enumerate(names):
#     plt.text(rhos_traj[i], thetas_traj[i], name, fontsize=10, ha='left', va='bottom', color='blue')

plt.xlabel('Rho', fontsize=18)
plt.ylabel('Theta', fontsize=18)
plt.title('Contrails and aircraft within the domains boundaries', fontsize=20)
plt.grid(True)
plt.legend()
# plt.savefig('Domains_boundaries.png')
plt.show()

# %%
# Plot the current suspects on Earth

color_cycle =  plt.cm.tab10.colors
trajectories = []
fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection=PlateCarree()))

t2 = fq.query(f"callsign in {suspect}")

with plt.style.context("traffic"):
    for j, vol in enumerate(t2):
        label = vol.callsign
        color = color_cycle[j % len(color_cycle)]
        line, *_ = vol.plot(ax, color=color, linewidth=2)
        vol.at().plot(ax,  color=color, text_kw=dict(font="Garuda", fontsize=10) )
        trajectories.append(line)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks(np.arange(round(west), round(east), 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(round(south), round(north), 2), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_xlim(west-1, east+1)
    ax.set_ylim(south-1, north+1)
    ax.set_title('{0}'.format(dt))
    im = ax.imshow(mask, extent=[west, east, south, north], cmap='gray', alpha=0.5)

plt.show()

# %%
# Filter according to direction

fq2 = fq.query(f"callsign in {suspect}")

final_suspect = []
list_flight = []
for flight in fq2:

    list_flight.append(flight.callsign)

    lon_i = flight.at_ratio(0.8)['longitude']
    lon_f = flight.at_ratio(1)['longitude']
    lat_i = flight.at_ratio(0.8)['latitude']
    lat_f = flight.at_ratio(1)['latitude']

    x_i = 0 + 256 * (lon_i - west) / (east - west)
    y_i =  256 - (0 + 256 * (lat_i - south) / (north - south))

    x_f = 0 + 256 * (lon_f - west) / (east - west)
    y_f = 256 - (0 + 256 * (lat_f - south) / (north - south))

    for k_c, v_c in dico_contrail.items():
        rho_v, theta_v, list_contrail = dico_traj[flight.callsign]
        if k_c in list_contrail: 

            d0 = np.sqrt( (x_i - v_c[2][0])**2 + (y_i - v_c[2][1])**2)
            d1 = np.sqrt( (x_f - v_c[2][0])**2 + (y_f - v_c[2][1])**2)
            d2 = np.sqrt( (x_i - v_c[2][2])**2 + (y_i - v_c[2][3])**2)
            d3 = np.sqrt( (x_f - v_c[2][2])**2 + (y_f - v_c[2][3])**2)

            if (d0 <= d1) and (d2 <= d3):
                final_suspect.append(flight.callsign)
                break

print("Final suspects are", final_suspect)

# %%
# Plot suspects and not suspects

t_suspect = fq2.query(f"callsign in {final_suspect}")

fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection=PlateCarree()))

with plt.style.context("traffic"):
    for j, vol in enumerate(fq2):
        #label = 'FL{:3d}'.format((int)(vol.data['altitude'].values[-1] / 100))
        color = color_cycle[j % len(color_cycle)]
        line, *_ = vol.plot(ax, color=color, linewidth=2)
        for i in range(10):
            try:
                vol.shorten(minutes=10*i).at().plot(ax,  color=color, text_kw=dict(font="Garuda", fontsize=10, text=5*i))
            except:
                # got multiple values for keyword argument 'text'
                pass
        last_lon = vol.data['longitude'].values[-1]
        last_lat = vol.data['latitude'].values[-1]
        #ax.text(last_lon, last_lat, label, ha='left', va='center', fontsize=12)
        if vol.callsign not in final_suspect:
            circle = plt.Circle((last_lon, last_lat), 0.1, color='teal')
            ax.add_patch(circle)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks(np.arange(round(west), round(east), 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(round(south), round(north), 2), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_xlim(west-1, east+1)
    ax.set_ylim(south-1, north+1)
    ax.set_title('Trajectories before the direction filter', fontsize=24)

    # Add wind
    # ax.barbs(longitude[::2], latitude[::2], u_wind[0,selected_level,::2,::2], v_wind[0,selected_level,::2,::2], alpha=0.5, transform=ccrs.PlateCarree())
    # ax.quiver(longitude[::2], latitude[::2], u_wind[0,selected_level,::2,::2], v_wind[0,selected_level,::2,::2], uv_wind[0,selected_level,::2,::2], cmap='coolwarm', alpha=1.0, transform=ccrs.PlateCarree())

    # Plot contrails
    colors = [(1, 1, 1, 0), (1, 0, 0, 1)]  # From transparent to red
    cmap = LinearSegmentedColormap.from_list('custom_reds', colors, N=256)
    im = ax.imshow(mask, extent=[west, east, south, north], cmap=cmap, zorder=1)

plt.show()


# %%
# Final image

fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection=PlateCarree()))

with plt.style.context("traffic"):
    for j, vol in enumerate(t_suspect):
        label = vol.callsign
        color = color_cycle[j % len(color_cycle)]
        line, *_ = vol.plot(ax, color=color, linewidth=2)
        for i in range(10):
            try:
                vol.shorten(minutes=10*i).at().plot(ax,  color=color, text_kw=dict(font="Garuda", fontsize=10, text=5*i))
            except:
                # got multiple values for keyword argument 'text'
                pass
        # ax.text(vol.data['longitude'].values[-1], vol.data['latitude'].values[-1], label, ha='left', va='center', fontsize=12)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks(np.arange(round(west), round(east), 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(round(south), round(north), 2), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_xlim(west-1, east+1)
    ax.set_ylim(south-1, north+1)
    ax.set_title('West Coast at the USA and Mexico border: {0}'.format(dt), fontsize=24)

    # Add wind
    ax.barbs(longitude[::2], latitude[::2], u_wind[0,selected_level,::2,::2], v_wind[0,selected_level,::2,::2], alpha=0.5, transform=ccrs.PlateCarree())
    # ax.quiver(longitude[::2], latitude[::2], u_wind[0,selected_level,::2,::2], v_wind[0,selected_level,::2,::2], uv_wind[0,selected_level,::2,::2], cmap='coolwarm', alpha=1.0, transform=ccrs.PlateCarree())

    # Plot contrails
    colors = [(1, 1, 1, 0), (1, 0, 0, 1)]  # From transparent to red
    cmap = LinearSegmentedColormap.from_list('custom_reds', colors, N=256)
    im = ax.imshow(mask, extent=[west, east, south, north], cmap=cmap, zorder=1)

plt.show()

#%%
