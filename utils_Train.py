
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

import statsmodels.api as sm
from statsmodels.formula.api import glm

from feature_engine.selection import SmartCorrelatedSelection
from scipy.stats import ks_2samp, ttest_ind
import infoselect as inf

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
import os

def split_5_fold_train_test(datos, fold):
    # Assure datos are randomized by rows before entering here
    datos_list = np.array_split(datos, 5)

    datos_test = datos_list[fold]
    # Combine the remaining parts for training data
    remaining_indices = [i for i in range(5) if i != fold]
    datos_train = pd.concat([datos_list[i] for i in remaining_indices], axis=0)

    print('[INFO] Shape de los datos de entrenamiento: ' + str(datos_train.shape))
    print('[INFO] Shape de los datos de prueba: ' + str(datos_test.shape))

    return datos_train, datos_test

def check_split(datos_list, datos_validation, datos_test):

    va_test_check = not(datos_test.equals(datos_validation))
    check_val_train, check_test_train = [], []
    for dataframe in datos_list[2:10]:
        check_val_train.append(not(dataframe.equals(datos_validation)))
        check_test_train.append(not(dataframe.equals(datos_test)))

    return va_test_check and all(check_val_train) and all(check_test_train)

def split_8_1_1(datos, fold):

    # divido los datos en bloques de 10; estńa randomizados por filas antes de entrar aqui
    datos_list = np.array_split(datos, 10)

    datos_test = datos_list[fold]
    if fold == 0:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[2:10]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 1:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[fold+2:10]+[datos_list[fold-1]]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 8:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[0:8]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 9:
        datos_validation = datos_list[0]
        datos_list_check = datos_list[1:9]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    else:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[fold+2:10]+datos_list[0:fold]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')

    print('[INFO] Shape de los datos de entrenamineto ' + str(datos_train.values.shape))

    return datos_train, datos_validation, datos_test

def outlier_flattening(datos_train, datos_test):
    datos_train_flat = datos_train.copy()
    datos_test_flat = datos_test.copy()

    for col in datos_train.columns:
        if col == 'sexo':
            continue
        else:
            percentiles = datos_train[col].quantile([0.025, 0.975]).values
            datos_train_flat[col] = np.clip(datos_train[col], percentiles[0], percentiles[1])
            datos_test_flat[col] = np.clip(datos_test[col], percentiles[0], percentiles[1])

    return datos_train_flat, datos_test_flat

def normalize_data_min_max(datos_train, datos_test, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_test = scaler.transform(datos_test)

    # Save the scaler to a file using pickle
    # with open('scaler.pkl', 'wb') as file:
    #     pickle.dump(scaler, file)

    return datos_train, datos_test

def standardize_data(datos_train, datos_test):
    columns = datos_train.columns

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit to the training data and transform both training and test data
    datos_train_scaled = scaler.fit_transform(datos_train)
    datos_test_scaled = scaler.transform(datos_test)

    # Convert numpy arrays back to pandas DataFrames, retaining original column names and indices
    datos_train_scaled = pd.DataFrame(datos_train_scaled, columns=columns)
    datos_test_scaled = pd.DataFrame(datos_test_scaled, columns=columns)

    return datos_train_scaled, datos_test_scaled

def feature_selection(data_train, data_val, data_test, ages_train, n_features):

    # select 10 percent best
    sel_2 = SelectPercentile(mutual_info_regression, percentile=30)
    data_train = sel_2.fit_transform(data_train, ages_train)
    data_val = sel_2.transform(data_val)
    data_test = sel_2.transform(data_test)

    data_train = pd.DataFrame(data_train)
    data_train.columns = sel_2.get_feature_names_out()
    data_val = pd.DataFrame(data_val)
    data_val.columns = sel_2.get_feature_names_out()
    data_test = pd.DataFrame(data_test)
    data_test.columns = sel_2.get_feature_names_out()

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    # Suppress ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Erase correlated feautures
    tr = SmartCorrelatedSelection(variables=None, method="pearson", threshold=0.8, missing_values="raise",
                                  selection_method="model_performance", estimator=MLPRegressor(max_iter=100, early_stopping=True, validation_fraction=0.1),
                                  scoring='neg_mean_absolute_error',  cv=3)

    # dcf = SmartCorrelatedSelection(threshold=0.80, method='pearson', selection_method="variance")
    data_train = tr.fit_transform(data_train, ages_train)
    data_val = tr.transform(data_val)
    data_test = tr.transform(data_test)

    # more MI selection
    gmm = inf.get_gmm(data_train.values, ages_train)
    select = inf.SelectVars(gmm, selection_mode='forward')
    select.fit(data_train.values, ages_train, verbose=False)

    # print(select.get_info())
    # select.plot_mi()
    # select.plot_delta()

    data_train_filtered = select.transform(data_train.values, rd=n_features)
    data_val_filtered = select.transform(data_val.values, rd=n_features)
    data_test_filtered = select.transform(data_test.values, rd=n_features)

    # data_train_filtered = data_train
    # data_test_filtered = data_test
    # features_names = sel_2.get_feature_names_out()

    indices = select.feat_hist[n_features]
    names_list = data_train.columns.tolist()
    features_names = [names_list[i] for i in indices]

    print("nº de features final: "+str(len(features_names)))

    return data_train_filtered, data_val_filtered, data_test_filtered, features_names

def define_lists_cnn():

    # defino listas para guardar los resultados y un dataframe # tab_CNN
    MAE_list_train_tab_CNN, MAE_list_train_unbiased_tab_CNN, r_list_train_tab_CNN, r_list_train_unbiased_tab_CNN, rs_BAG_train_tab_CNN, \
    rs_BAG_train_unbiased_tab_CNN, alfas_tab_CNN, betas_tab_CNN = [], [], [], [], [], [], [], []
    BAG_ChronoAge_df_tab_CNN = pd.DataFrame()

    listas_tab_CNN = [MAE_list_train_tab_CNN, MAE_list_train_unbiased_tab_CNN, r_list_train_tab_CNN,
                  r_list_train_unbiased_tab_CNN, rs_BAG_train_tab_CNN, rs_BAG_train_unbiased_tab_CNN, alfas_tab_CNN,
                  betas_tab_CNN, BAG_ChronoAge_df_tab_CNN, 'tab_CNN']

    return listas_tab_CNN

def execute_in_val_and_test_NN(data_train_filtered, edades_train, data_val_filtered, edades_val, data_test_filtered, edades_test, lista, regresor, n_features, save_dir, fold):

    # identifico en método de regresión
    regresor_used = lista[9]

    # hago el entrenamiento sobre todos los datos de entrenamiento
    regresor.fit(data_train_filtered, edades_train, data_val_filtered, edades_val, fold, epochs=500, lr=5e-4, weight_decay=1e-5)

    data_train_filtered = pd.DataFrame(data_train_filtered)
    data_train_filtered['Edades'] = edades_train
    data_val_filtered = pd.DataFrame(data_val_filtered)
    data_val_filtered['Edades'] = edades_val
    data_test_filtered = pd.DataFrame(data_test_filtered)
    data_test_filtered['Edades'] = edades_test

    data_test_filtered, data_cal_filtered = train_test_split(data_test_filtered, test_size=0.5, random_state=42)

    data_train_filtered.to_csv(os.path.join(save_dir, 'datos_train.csv'), index=False)
    data_val_filtered.to_csv(os.path.join(save_dir, 'datos_val.csv'),  index=False)
    data_test_filtered.to_csv(os.path.join(save_dir, 'datos_test.csv'), index=False)
    data_cal_filtered.to_csv(os.path.join(save_dir, 'datos_cal.csv'), index=False)

    edades_train = data_train_filtered['Edades'].values
    data_train_filtered = data_train_filtered.drop('Edades', axis=1)
    data_train_filtered = data_train_filtered.values
    edades_val = data_val_filtered['Edades'].values
    data_val_filtered = data_val_filtered.drop('Edades', axis=1)
    data_val_filtered = data_val_filtered.values
    edades_test = data_test_filtered['Edades'].values
    data_test_filtered = data_test_filtered.drop('Edades', axis=1)
    data_test_filtered = data_test_filtered.values
    edades_cal = data_cal_filtered['Edades'].values
    data_cal_filtered = data_cal_filtered.drop('Edades', axis=1)
    data_cal_filtered = data_cal_filtered.values

    ks_test = ks_2samp(edades_train, edades_cal)
    print('test de Kolmogorov-Smirnov para edades train-val')
    print('si es mayor de 0.05 no puedo descartar que sean iguales: '+str(ks_test[1]))

    regresor.calculate_calibration_constant(data_cal_filtered, edades_cal, alpha=0.01)

    pred_train_all = regresor.predict(data_train_filtered, apply_calibration=False)
    pred_train = pred_train_all['median_aleatory_epistemic']

    # Create DataFrame
    df_bias_correction = pd.DataFrame({
        'edades_train': edades_train,
        'pred_train': pred_train
    })

    df_bias_correction.to_csv(os.path.join(save_dir, 'DataFrame_bias_correction.csv'), index=False)

    # Hago la predicción de los casos de test sanos
    pred_test_all = regresor.predict(data_test_filtered, apply_calibration=False)

    pred_test_84 = pred_test_all['0.840_aleatory_epistemic']
    pred_test = pred_test_all['median_aleatory_epistemic']
    pred_test_16 = pred_test_all['0.160_aleatory_epistemic']

    # Calculo BAG sanos val & test
    BAG_test_sanos = pred_test - edades_test

    # calculo MAE, MAPE y r test
    MAE_biased_test = mean_absolute_error(edades_test, pred_test)
    MAPE_biased_test = mean_absolute_percentage_error(edades_test, pred_test)
    r_squared = r2_score(edades_test, pred_test)
    r_biased_test = stats.pearsonr(edades_test, pred_test)[0]
    r_bag_real_biased_test = stats.pearsonr(BAG_test_sanos, edades_test)[0]

    # Calculo r MAE para test
    print('----------- ' + regresor_used + ' r & MAE test biased -------------')
    print('MAE test: ' + str(MAE_biased_test))
    print('MAPE test: ' + str(MAPE_biased_test))
    print('r test: ' + str(r_biased_test))
    print('R2 test: ' + str(r_squared))

    # calculo r biased test
    print('--------- ' + regresor_used + ' Correlación BAG edad real test -------------')
    print('r BAG-edad real test biased: ' + str(r_bag_real_biased_test))
    print('')

    # Figura concordancia entre predichas y reales con reg lineal
    plt.figure(figsize=(8, 6))
    plt.scatter(edades_test, pred_test, color='blue', label='Predictions')
    plt.plot([edades_test.min(), edades_test.max()], [edades_test.min(), edades_test.max()], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Real Age')
    plt.ylabel('Predicted Age')
    plt.title('Predicted Age vs. Real Age')
    plt.legend()

    # Annotate MAE, Pearson correlation r, and R² in the plot
    textstr = '\n'.join((
        f'MAE: {MAE_biased_test:.2f}',
        f'Pearson r: {r_biased_test:.2f}',
        f'R²: {r_squared:.2f}'))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # plt.show()

    plt.savefig('Model_PredAge_vs_Age_fold_'+str(fold)+'.svg')

    MAEs_and_rs_test = pd.DataFrame(list(zip([MAE_biased_test], [r_biased_test], [r_bag_real_biased_test])),
                                    columns=['MAE_biased_test', 'r_biased_test', 'r_bag_real_biased_test'])

    # save the model to disk
    filename = os.path.join(save_dir, 'MCCQMLP_nfeats_' + str(n_features) + '_fold_'+ str(fold) +'.pkl')
    pickle.dump(regresor, open(filename, 'wb'))

    # results = permutation_importance(regresor, data_train_filtered, edades_train, scoring='neg_mean_absolute_error', n_jobs=-1)

    return MAEs_and_rs_test

def fit_glm_and_get_coef(data, formula, coef_name):
    model = glm(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    return model.params[coef_name]

def fit_glm_and_get_all_coef(data, formula):
    model = glm(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    return model.params

def perform_tests(group0, group1):
    stat, p_value = ttest_ind(group0, group1)
    return stat

def rain_cloud_plot(df_prev_copy):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['Group'].unique()

    for group in groups:
        x_values[group] = df[df['Group'] == group]['BrainPAD'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    start_blue_colors = '#91bec8'
    end_blue_colors = '#0e4e6e'

    # Generate gradient colors for different groups
    start_color_rgb = hex_to_rgb(start_blue_colors)
    end_color_rgb = hex_to_rgb(end_blue_colors)
    palette_1 = interpolate_colors(start_color_rgb, end_color_rgb, len(groups))

    start_blue_colors = '#6fa4b7'
    end_blue_colors = '#146e9b'
    palette_2 = interpolate_colors(hex_to_rgb(start_blue_colors), hex_to_rgb(end_blue_colors), len(groups))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with axes swapped
    violins = sns.violinplot(data=df, y='BrainPAD', x='Group', hue='Group', palette=palette_2, ax=ax, fill=False,
                             inner_kws=dict(  # Custom settings for the inner boxplot
                                 box_width=0.2,  # Custom width for the inner boxplot
                                 whis_width=2,  # Custom whisker width
                                 color="black",  # Custom color for the inner boxplot and whiskers
                             ),
                             legend=False)

    # Apply alpha to the violin plot elements
    for violin in violins.collections:
        violin.set_alpha(0.3)


    # Scatter plot on top of the violin plot with axes swapped
    for x, val, c in zip(xs, vals, palette_1):
        plt.scatter(x, val, alpha=0.4, color=c, edgecolor=c, zorder=1)

    # Customizing plot
    plt.ylabel('BrainPAD', fontweight='bold')
    plt.xlabel('Group', fontweight='bold')
    plt.title('Raincloud Plot with Violin for BrainPAD by Group (Axes Switched)', fontweight='bold')

    plt.show()

def rain_cloud_plot_II(df_prev_copy):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['Group'].unique()

    for group in groups:
        x_values[group] = df[df['Group'] == group]['BrainPAD'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    start_blue_colors = '#91bec8'
    end_blue_colors = '#0e4e6e'

    # Generate gradient colors for different groups
    start_color_rgb = hex_to_rgb(start_blue_colors)
    end_color_rgb = hex_to_rgb(end_blue_colors)
    palette_1 = interpolate_colors(start_color_rgb, end_color_rgb, len(groups))

    start_blue_colors = '#6fa4b7'
    end_blue_colors = '#146e9b'
    palette_2 = interpolate_colors(hex_to_rgb(start_blue_colors), hex_to_rgb(end_blue_colors), len(groups))

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with no inner fill but outlined box
    sns.violinplot(
        data=df,
        x='BrainPAD',
        y='Group',
        inner=None,  # No inner plots as we will overlay the boxplot
        palette=palette_2,  # Color palette for violins
        linewidth=1.5,  # Line width for the violin outline
        ax=ax
    )

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    sns.boxplot(
        data=df,
        x='BrainPAD',
        y='Group',
        width=0.2,  # Narrower box width
        boxprops=dict(facecolor='none', edgecolor=palette_2, linewidth=1),  # Only outline for the box, no fill
        whiskerprops=dict(linewidth=1),  # Thicker whiskers
        capprops=dict(linewidth=1),  # Thicker caps
        medianprops=dict(linewidth=1, color=palette_2),  # Thicker median line
        showfliers=False,  # Optional: Hide outliers
        ax=ax
    )

    # Customizing plot
    plt.ylabel('Group', fontweight='bold')
    plt.xlabel('BrainPAD', fontweight='bold')
    plt.title('Violin Plot with Outline Boxplot for BrainPAD by Group', fontweight='bold')

    plt.show()

def rain_cloud_plot_III(df_prev_copy):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    df = df_prev_copy

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['Group'].unique()

    for group in groups:
        x_values[group] = df[df['Group'] == group]['brainPAD_standardized'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    grey_colors_light = '#B4BBBB'
    gray_colors_dark = '#858E8D'
    blue_colors_light = '#80AFEC'
    blue_colors_dark = '#1D5FB5'
    red_colors_light = '#BF6673'
    red_colors_dark = '#C42840'
    orange_colors_light = '#EC9468'
    orange_colors_dark = '#DC5614'

    # Generate gradient colors for different groups
    palette_1 = interpolate_colors(hex_to_rgb(grey_colors_light), hex_to_rgb(red_colors_light), len(groups))

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with no fill and switched axes (Group on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='Group',
        y='brainPAD_standardized',
        palette=[gray_colors_dark, red_colors_dark],  # Color palette for violin outlines
        linewidth=3.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(groups):
        sns.boxplot(
            data=df[df['Group'] == group],
            x='Group',
            y='brainPAD_standardized',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='Group',
        y='brainPAD_standardized',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=[grey_colors_light, red_colors_light],  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(groups):
        group_mean = df[df['Group'] == group]['brainPAD_standardized'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('Group', fontweight='bold')
    plt.ylabel('brainPAD_standardized', fontweight='bold')
    plt.title('Violin Plot', fontweight='bold')

    plt.show()

def generate_generic_data():
    np.random.seed(42)  # Set seed for reproducibility
    n = 100  # Number of samples per group
    groups = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

    data = {
        'Group': np.repeat(groups, n),
        'BrainPAD': np.concatenate([
            np.random.normal(0, 1, n),  # Group 1
            np.random.normal(1, 1, n),  # Group 2
            np.random.normal(2, 1, n),  # Group 3
            np.random.normal(3, 1, n)  # Group 4
        ])
    }

    return pd.DataFrame(data)

def rain_cloud_plot_V(df):

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Define colors for the groups
    grey_colors_light = '#B4BBBB'
    gray_colors_dark = '#858E8D'

    blue_colors_light = '#70B7CE'
    blue_colors_dark = '#348AA7'

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palettes for the groups
    palette_red_light = ['#CE9AA3', '#D07986', '#DC576B']
    palette_red_dark = ['#B67781', '#B65362', '#B52F43']

    palette_orange_light = ['#E2A98C', '#EC9468', '#F17D47']
    palette_orange_dark = ['#CC8E79', '#DA7949', '#DC5614']

    # Violin plot with no fill and switched axes (pliks18TH on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='pliks18TH',
        y='brainPAD_standardized',
        palette=[gray_colors_dark] + palette_red_dark,  # Color palette for violin outlines
        linewidth=2.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    palette_1 = [grey_colors_light] + palette_red_light

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(df['pliks18TH'].unique()):
        sns.boxplot(
            data=df[df['pliks18TH'] == group],
            x='pliks18TH',
            y='brainPAD_standardized',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='pliks18TH',
        y='brainPAD_standardized',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=palette_1,  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(df['pliks18TH'].unique()):
        group_mean = df[df['pliks18TH'] == group]['brainPAD_standardized'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('pliks18TH', fontweight='bold')
    plt.ylabel('brainPAD_standardized', fontweight='bold')
    plt.title('Violin Plot for standardized brainPAD by group', fontweight='bold')

    plt.show()

def rain_cloud_plot_IV(df_prev_copy):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    df = df_prev_copy

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['grupo'].unique()

    for group in groups:
        x_values[group] = df[df['grupo'] == group]['Delta_BAG_standardized'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    grey_colors_light = '#B4BBBB'
    gray_colors_dark = '#858E8D'
    blue_colors_light = '#D8BFD8'
    blue_colors_dark = '#7E6A9C'
    red_colors_light = '#BF6673'
    red_colors_dark = '#C42840'
    orange_colors_light = '#A3B583'
    orange_colors_dark = '#556B2F'

    # Generate gradient colors for different groups
    palette_1 = interpolate_colors(hex_to_rgb(blue_colors_light), hex_to_rgb(orange_colors_light), len(groups))

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['grupo'] = df['grupo'].replace('Crazy', 'PE_1-3')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with no fill and switched axes (Group on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='grupo',
        y='Delta_BAG_standardized',
        palette=[blue_colors_dark, orange_colors_dark],  # Color palette for violin outlines
        linewidth=3.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(groups):
        sns.boxplot(
            data=df[df['grupo'] == group],
            x='grupo',
            y='Delta_BAG_standardized',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='grupo',
        y='Delta_BAG_standardized',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=[blue_colors_light, orange_colors_light],  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(groups):
        group_mean = df[df['grupo'] == group]['Delta_BAG_standardized'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('grupo', fontweight='bold')
    plt.ylabel('Delta_BAG_standardized', fontweight='bold')
    plt.title('Violin Plot', fontweight='bold')

    plt.show()

def rain_cloud_plot_VI(df_prev_copy):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    df = df_prev_copy
    df['grupo_cat'] = df['grupo_cat'].replace('Enfermos', 'PEs')
    df['grupo_cat'] = df['grupo_cat'].replace('Controles', 'Controls')

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['grupo_cat'].unique()

    for group in groups:
        x_values[group] = df[df['grupo_cat'] == group]['DeltaBrainPAD'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    purple_colors_light = '#C59CDC'  # Light purple
    purple_colors_dark = '#6A3D9A'  # Dark purple
    green_colors_light = '#A6E6A1'  # Light green
    green_colors_dark = '#228B22'  # Dark green

    # Generate gradient colors for different groups
    palette_1 = interpolate_colors(hex_to_rgb(purple_colors_light), hex_to_rgb(green_colors_light), len(groups))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with no fill and switched axes (Group on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='grupo_cat',
        y='DeltaBrainPAD',
        palette=[purple_colors_dark, green_colors_dark],  # Color palette for violin outlines
        linewidth=3.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(groups):
        sns.boxplot(
            data=df[df['grupo_cat'] == group],
            x='grupo_cat',
            y='DeltaBrainPAD',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='grupo_cat',
        y='DeltaBrainPAD',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=[purple_colors_light, green_colors_light],  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(groups):
        group_mean = df[df['grupo_cat'] == group]['DeltaBrainPAD'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('grupo_cat', fontweight='bold')
    plt.ylabel('DeltaBrainPAD', fontweight='bold')
    plt.title('Violin Plot', fontweight='bold')

    plt.show()

def rain_cloud_plot_VIII(df):

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Define colors for the groups
    purple_colors_light = '#C59CDC'  # Light purple
    purple_colors_dark = '#6A3D9A'  # Dark purple

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palettes for the groups
    palette_green_light = ['#A6E6A1', '#7AC96B', '#4CA743']  # Shades of green transitioning from light
    palette_green_dark = ['#67C477', '#3E9E31', '#228B22']  # Darker shades of green

    # Violin plot with no fill and switched axes (pliks18TH on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='Grupo_ordinal',
        y='DeltaBrainPAD',
        palette= [purple_colors_dark] + ['#67C477'] + ['#3E9E31', '#228B22'],  # Color palette for violin outlines
        linewidth=2.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    palette_1 =[purple_colors_light] + ['#A6E6A1'] + ['#7AC96B', '#4CA743']

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(sorted(df['Grupo_ordinal'].unique())):
        sns.boxplot(
            data=df[df['Grupo_ordinal'] == group],
            x='Grupo_ordinal',
            y='DeltaBrainPAD',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='Grupo_ordinal',
        y='DeltaBrainPAD',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=palette_1,  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(sorted(df['Grupo_ordinal'].unique())):
        group_mean = df[df['Grupo_ordinal'] == group]['DeltaBrainPAD'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('Grupo_ordinal', fontweight='bold')
    plt.ylabel('DeltaBrainPAD', fontweight='bold')
    plt.title('Violin Plot for brainPAD_standardized by grupo', fontweight='bold')

    plt.show()

def rain_cloud_plot_VII(df):

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Define colors for the groups
    purple_colors_light = '#C59CDC'  # Light purple
    purple_colors_dark = '#6A3D9A'  # Dark purple

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palettes for the groups
    palette_green_light = ['#A6E6A1', '#7AC96B', '#4CA743']  # Shades of green transitioning from light
    palette_green_dark = ['#67C477', '#3E9E31', '#228B22']  # Darker shades of green

    # Violin plot with no fill and switched axes (pliks18TH on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='Grupo_ordinal',
        y='DeltaBrainPAD',
        palette= ['#67C477'] + [purple_colors_dark] + ['#3E9E31', '#228B22'],  # Color palette for violin outlines
        linewidth=2.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    palette_1 = ['#A6E6A1'] + [purple_colors_light] + ['#7AC96B', '#4CA743']

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(sorted(df['Grupo_ordinal'].unique())):
        sns.boxplot(
            data=df[df['Grupo_ordinal'] == group],
            x='Grupo_ordinal',
            y='DeltaBrainPAD',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='Grupo_ordinal',
        y='DeltaBrainPAD',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=palette_1,  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(sorted(df['Grupo_ordinal'].unique())):
        group_mean = df[df['Grupo_ordinal'] == group]['DeltaBrainPAD'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('Grupo_ordinal', fontweight='bold')
    plt.ylabel('DeltaBrainPAD', fontweight='bold')
    plt.title('Violin Plot for brainPAD_standardized by grupo', fontweight='bold')

    plt.show()

def rain_cloud_plot_colors(df):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Define colors for the groups
    grey_colors_light = '#B4BBBB'
    gray_colors_dark = '#858E8D'

    blue_colors_light = '#70B7CE'
    blue_colors_dark = '#348AA7'

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palettes for the groups
    palette_red_light = ['#CE9AA3', '#D07986', '#DC576B']
    palette_red_dark = ['#B67781', '#B65362', '#B52F43']

    palette_orange_light = ['#E2A98C', '#EC9468', '#F17D47']
    palette_orange_dark = ['#CC8E79', '#DA7949', '#DC5614']

    # Violin plot with no fill and switched axes (pliks18TH on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='pliks18TH',
        y='brainPAD_standardized',
        palette=[gray_colors_dark] + palette_red_dark,  # Color palette for violin outlines
        linewidth=2.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    palette_1 = [grey_colors_light] + palette_red_light

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(df['pliks18TH'].unique()):
        sns.boxplot(
            data=df[df['pliks18TH'] == group],
            x='pliks18TH',
            y='brainPAD_standardized',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    # Determine the last group and top 10 highest brainPAD_standardized
    last_group = df['pliks18TH'].unique()[-1]
    last_group_top_7 = df[df['pliks18TH'] == last_group].nlargest(7, 'brainPAD_standardized')

    sns.stripplot(
        data=df,
        x='pliks18TH',
        y='brainPAD_standardized',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=palette_1,  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    for ID in last_group_top_7['ID'].values.tolist():
        print(ID)

    print(last_group_top_7)

    # Highlight top 10 IDs in blue for the last group
    ax.scatter(
        [last_group] * len(last_group_top_7),  # Position on the x-axis for the last group
        last_group_top_7['brainPAD_standardized'],  # Position on the y-axis
        color=blue_colors_dark,  # Blue color for the top 10 points
        s=70,  # Size of the points
        zorder=3, # Ensure it appears on top of other elements
        alpha=0.4
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(df['pliks18TH'].unique()):
        group_mean = df[df['pliks18TH'] == group]['brainPAD_standardized'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('pliks18TH', fontweight='bold')
    plt.ylabel('brainPAD_standardized', fontweight='bold')
    plt.title('Violin Plot for brainPAD_standardized by group', fontweight='bold')

    plt.show()

def calculate_cohen_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean2 - mean1) / pooled_std

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def age_vs_predicted_age_figure(edades_test, pred_test):

    # calculo MAE, MAPE y r test
    MAE_biased_test = mean_absolute_error(edades_test, pred_test)
    r_squared = r2_score(edades_test, pred_test)
    r_biased_test = stats.pearsonr(edades_test, pred_test)[0]

    # Figura concordancia entre predichas y reales con reg lineal
    plt.figure(figsize=(8, 8))
    plt.scatter(edades_test, pred_test, color='blue', label='Predictions')
    plt.plot([10, 100], [10, 100], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Real Age')
    plt.ylabel('Predicted Age')
    plt.title('Predicted Age vs. Real Age')

    # Set x and y axis limits
    plt.xlim(5, 100)
    plt.ylim(5, 100)

    # Ensure x and y axes have the same scale
    plt.gca().set_aspect('equal', adjustable='box')

    # Annotate MAE, Pearson correlation r, and R² in the plot
    textstr = '\n'.join((
        f'MAE: {MAE_biased_test:.2f}',
        f'Pearson r: {r_biased_test:.2f}',
        f'R²: {r_squared:.2f}'))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.show()

def model_evaluation_modified(X_test, results):

    X_test_df = pd.read_csv(X_test)

    etiv = X_test_df['eTIV'].values.tolist()
    edades_test = X_test_df['Age'].values.tolist()

    # Load Model
    file_path = ('/home/rafa/PycharmProjects/ALSPAC_BA/Model/SimpleMLP_nfeats_100_fold_0.pkl')
    with open(file_path, 'rb') as file:
        regresor = pickle.load(file)

    # predict age
    prediction = regresor.predict(X_test_df.iloc[:, :-2].values)

    # bias correction
    df_bias_correction = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Model/DataFrame_bias_correction.csv')

    model = LinearRegression()
    model.fit(df_bias_correction[['edades_train']], df_bias_correction['pred_train'])

    slope = model.coef_[0]
    intercept = model.intercept_

    # Age bias correction
    results['pred_Age_c'] = (prediction - intercept) / slope
    results['pred_Age'] = prediction
    results['BrainPAD'] = results['pred_Age_c'] - edades_test
    results['eTIV'] = etiv

    return results

def model_evaluation(X_train, X_test, results, features):
    etiv = X_test['eTIV'].values.tolist()

    edades_train = X_train['Edad'].values
    edades_test = results['Edad'].values

    X_train = X_train[features]
    X_test = X_test[features]

    # aplico la eliminación de outliers
    X_train, X_test = outlier_flattening(X_train, X_test)

    # 3.- normalizo los datos OJO LA NORMALIZACION QUE CON Z NORM O CON 0-1 PUEDE VARIAR EL RESULTADO BASTANTE!
    X_train, X_test = normalize_data_min_max(X_train, X_test, (-1, 1))

    X_train_df = pd.DataFrame(X_train)
    X_train_df.columns = features
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = features
    X_test_df['Edad'] = edades_test
    X_train_df['Edad'] = edades_train

    X_test_df['eTIV'] = etiv

    X_test_df.to_csv('data_test_ALSPAC_II.csv', index=False)

    file_path = ('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/SimpleMLP_nfeats_100_fold_0.pkl')
    with open(file_path, 'rb') as file:
        regresor = pickle.load(file)

    pred_test_median = regresor.predict(X_test)

    # bias correction
    df_bias_correction = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/DataFrame_bias_correction.csv')

    model = LinearRegression()
    model.fit(df_bias_correction[['edades_train']], df_bias_correction['pred_train'])

    slope = model.coef_[0]
    intercept = model.intercept_

    results['pred_Edad_c'] = (pred_test_median - intercept) / slope
    results['pred_Edad'] = pred_test_median
    results['BrainPAD'] = results['pred_Edad_c'] - edades_test
    results['eTIV'] = etiv

    results.rename(columns={'sexo(M=1;F=0)': 'sexo'}, inplace=True)

    return results

def benjamini_hochberg_correction(p_values):
    n = len(p_values)
    sorted_p_values = np.array(sorted(p_values))
    ranks = np.arange(1, n+1)

    # Calculate the cumulative minimum of the adjusted p-values in reverse
    adjusted_p_values = np.minimum.accumulate((sorted_p_values * n) / ranks)[::-1]

    # Reverse back to original order
    reverse_indices = np.argsort(p_values)
    return adjusted_p_values[reverse_indices]

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, r, r2

def standardize_with_control(control_df, target_df, gender_col, value_col, scaler_men, scaler_women):
    """
    Standardize a specified column in the target DataFrame using scalers fitted on the control DataFrame.

    Args:
    - control_df (pd.DataFrame): The DataFrame used to fit the scalers.
    - target_df (pd.DataFrame): The DataFrame to apply the standardization.
    - gender_col (str): Column indicating gender (1 for men, 0 for women).
    - value_col (str): Column to standardize.
    - scaler_men (StandardScaler): Scaler for men.
    - scaler_women (StandardScaler): Scaler for women.

    Returns:
    - pd.DataFrame: A DataFrame with standardized values added as a new column.
    """
    # Split control group by gender
    control_men = control_df[control_df[gender_col] == 1].copy()
    control_women = control_df[control_df[gender_col] == 0].copy()

    # Fit the scalers on the control group
    scaler_men.fit(control_men[[value_col]])
    scaler_women.fit(control_women[[value_col]])

    # Split target DataFrame by gender
    target_men = target_df[target_df[gender_col] == 1].copy()
    target_women = target_df[target_df[gender_col] == 0].copy()

    # Apply the scalers to the target DataFrame
    target_men['brainPAD_standardized'] = scaler_men.transform(target_men[[value_col]])
    target_women['brainPAD_standardized'] = scaler_women.transform(target_women[[value_col]])

    return pd.concat([target_men, target_women], axis=0)


