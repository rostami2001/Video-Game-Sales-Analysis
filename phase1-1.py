from cmath import nan
from tabulate import tabulate
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from seaborn import histplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
import numpy as np
import statistics
import re
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import ttest_ind
from colorama import Fore, Back, Style
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import LabelEncoder

def convet_to_numeric(data_attribute):
    average_scores = []
    for value in data_attribute:
        total_score = 0
        count = 0
        for item in eval(value):
            score_str = item['Platform Metascore']
            if score_str.isdigit():
                total_score += int(score_str)
                count += 1
            else:
                total_score = 0
        if count > 0:
            # print(total_score/count)
            average_scores.append(total_score / count)
        else:
            average_scores.append(0)
    value_mapping = dict(zip(data_attribute, average_scores))

    mapped_datas = data_attribute.apply(lambda x: value_mapping.get(x, None))

    return value_mapping, mapped_datas


def Knowing_dataset(df):
    Attribute_list = []
    header = ['feature name', 'type', 'range', 'min', 'max', 'mean', 'median', 'mode', 'outlier']
    box_plots_columns = []
    box_plots_columns_title = []

    for column_name, column_type in df.dtypes.items():
        new_attribute = []
        new_attribute.append(column_name)  # feature name
        new_attribute.append(column_type)  # type

        # range & min & max & mean
        if column_type == 'int64' or column_type == 'float64':

            # df[column_name].dropna(inplace=True)

            min_value = df[column_name].min()
            max_value = df[column_name].max()
            mean_value = df[column_name].mean()
            range_value = f'({min_value}, {max_value})'
            median_value = df[column_name].median()
            mode_value = statistics.mode(df[column_name])

            new_attribute.append(range_value)
            new_attribute.append(min_value)
            new_attribute.append(max_value)
            new_attribute.append(mean_value)
            new_attribute.append(median_value)
            new_attribute.append(mode_value)

            box_plots_columns.append(df[column_name])
            box_plots_columns_title.append(column_name)

            boxplot_value = df.boxplot(column=column_name)
            plt.show()

            # Calculate outliers
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)][column_name]
            new_attribute.append(np.array(outliers))

        else:
            if df[column_name].dtype == 'object' and column_name != 'Platforms Info':
                new_attribute.append('-')
                new_attribute.append('-')
                new_attribute.append('-')
                new_attribute.append('-')
                new_attribute.append('-')
                new_attribute.append('-')
                new_attribute.append('-')

                # range_value = '-'
                # print(column_name)
            elif (column_name == 'Platforms Info'):
                value_mapping, mapped_datas = convet_to_numeric(df[column_name])
                min = np.array(mapped_datas).min()
                max = np.array(mapped_datas).max()
                mean = np.array(mapped_datas).mean()
                median = mapped_datas.median()
                mode = statistics.mode(mapped_datas)
                range_value = f'({min}, {max})'
                Q1 = mapped_datas.quantile(0.25)
                Q3 = mapped_datas.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = mapped_datas[(mapped_datas < lower_bound) | (mapped_datas > upper_bound)]

                new_attribute.append(range_value)
                new_attribute.append(min)
                new_attribute.append(max)
                new_attribute.append(mean)
                new_attribute.append(median)  # jahasho avaz kardam!
                new_attribute.append(mode)
                new_attribute.append(np.array(outliers))

                plt.title(column_name)
                plt.boxplot(np.array(mapped_datas))
                # plt.show()


            else:
                new_attribute.append('-')  # min
                new_attribute.append('-')  # max
                new_attribute.append('-')  # mean
                new_attribute.append('-')  # median
                new_attribute.append('-')  # mode (ezafe kardam!)

        Attribute_list.append(new_attribute)

    num_attr = len(box_plots_columns_title)
    num_cols = int(np.ceil(np.sqrt(num_attr)))
    fig, axes = plt.subplots(num_cols, num_cols, constrained_layout=True)
    plt.title("boxplot attributes")

    for i, ax in enumerate(axes.flat):
        if i < num_attr:
            ax.boxplot(box_plots_columns[i])
            ax.set_title(box_plots_columns_title[i])
            ax.axis('off')
        else:
            ax.axis('off')

    plt.show()
    print(tabulate(Attribute_list, headers=header, tablefmt="grid"))

    return df


def check_consistency(df):
    inconsistent_rows = df[df['release_date'] > df['last_update']]
    num_inconsistent = len(inconsistent_rows)
    num_total = len(df)
    consistency_ratio = 1 - (num_inconsistent / num_total)
    return consistency_ratio, num_inconsistent, inconsistent_rows


def validity_attr(df, attr):
    details = pd.read_csv('info.csv')
    row_index = details[details['name'] == attr].index[0]
    range_attr = details['range'][row_index]
    range_type_attr = details['range_type'][row_index]

    sum_consistency = 0

    if range_type_attr == 'range':

        range_attr_arr = range_attr.split(',')
        for datapoints in df[attr]:
            if (float(range_attr_arr[0]) <= datapoints <= float(range_attr_arr[1])):
                sum_consistency += 1

    elif range_type_attr == 'list':
        range_attr_arr = range_attr.split(',')
        for datapoints in df[attr]:
            if (str(datapoints) in range_attr_arr):
                sum_consistency += 1


    else:
        for datapoints in df[attr]:
            if re.match(range_attr, str(datapoints)):
                sum_consistency += 1

    return sum_consistency / len(df[attr])


def Data_quality_assessment(df):
    Attribute_list = []
    header = ['feature name', 'number of records', 'number of Null values', 'Accuracy', 'completness', 'validity',
              'currentness', 'consistency']

    for column_name, column_type in df.dtypes.items():
        new_attribute = []

        number_of_records = len(df[column_name])
        number_of_Null_values = df[column_name].isnull().sum()
        completness = 1 - (number_of_Null_values / number_of_records)
        consistency = '-'
        currentness = '-'
        validity = validity_attr(df, column_name)
        accuracy = f'{(validity * 100):.2f}%'

        new_attribute.append(column_name)  # feature name
        new_attribute.append(number_of_records)
        new_attribute.append(number_of_Null_values)  # number of Null values
        new_attribute.append(accuracy)
        new_attribute.append(completness)
        new_attribute.append(validity)
        new_attribute.append(currentness)
        new_attribute.append(consistency)

        Attribute_list.append(new_attribute)

    print(tabulate(Attribute_list, headers=header, tablefmt="grid"))


def combination_df(df1, df2):
    df1 = df1.rename(columns={'title': 'Title', 'genre': 'Genres', 'publisher': 'Publisher', 'developer': 'Developer'
        , 'critic_score': 'User Score', 'release_date': 'Release Date'})

    for col in df2.columns:
        if col not in df1.columns and col != 'Genres Splitted':
            df1[col] = df2[col]

    df1 = df1.drop_duplicates(subset=['Title'])
    for index, row in df1.iterrows():
        matching_row = df2[df2['Title'] == row['Title']]
        if matching_row is None:
            for col in df1.columns:
                if col in df2.columns and pd.isna(row[col]):
                    df1.at[index, col] = matching_row.iloc[0][col]
                elif col in df2.columns and row[col] != matching_row.iloc[0][col]:
                    if (col == 'release_date'):
                        if (pd.notna(row[col]) and pd.notna(matching_row.iloc[0][col])):
                            date_format = "%d/%m/%Y"
                            date1 = datetime.strptime(row[col], date_format)
                            date2 = datetime.strptime(matching_row.iloc[0][col], date_format)
                            if date1 < date2:
                                df1.at[index, col] = row[col]
                            else:
                                df1.at[index, col] = matching_row.iloc[0][col]
    # df1.to_csv('after_merge.csv', index=False)
    return df1


def handle_missing_data(df):
    for col in df.columns:
        first_missing_value = df[col].isnull().sum()
        completness = 1 - (first_missing_value / len(df[col]))
        if completness < 0.2:
            df = df.drop([col], axis=1)
            print("Droped column " + col + ": with " + str(first_missing_value) + " missing value & "
                  + str(completness) + " completness")
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    non_numerical_cols = df.select_dtypes(exclude=['number']).columns
    df[non_numerical_cols] = df[non_numerical_cols].fillna(df[non_numerical_cols].mode().iloc[0])

    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        print("No missing data found.")
        # df.to_csv('after_handle_missing_data.csv', index=False)
    else:
        print("Missing data exists. Please handle missing values.")
        print(missing_data)

    return df


def normalization(df, attr):
    for a in attr:
        # mean_value = df[attr].mean()
        # std_dev = df[attr].std()
        max_value = df[a].max()
        min_value = df[a].min()

        normal_value = f'{a}_normal'
        df[normal_value] = (df[a] - min_value) / (max_value - min_value)
    # df.to_csv('after_normalization.csv', index=False)
    print("normalization is done")


def evaluate_genre(genres, total_sales):
    if (genres == 'Action' or 'Shooter' or 'Action-Adventure' or 'Sports' or 'Role-Playing'
        or 'Simulation' or 'Racing' or 'Music' or 'Misc' or 'Fighting'
        or 'Platform' or 'Adventure' or 'Strategy' or 'Puzzle' or 'MMO'
        or 'Party' or 'Education' or 'Board Game' or 'Visual Novel' or 'Sandbox') and (total_sales >= 0.6):
        return '1'
    else:
        return '0'


def evaluate_console(console, total_sales):
    if (console == 'PS3' or 'PS2' or 'PS4' or 'X360' or 'PC'
        or 'PSP' or 'Wii' or 'PS' or 'DS' or '2600'
        or 'GBA' or 'NES' or 'XOne' or 'XB' or 'GEN'
        or 'DC' or 'N64' or 'SAT' or 'SNES' or 'GBC'
        or 'GC' or '3DS' or 'GB' or 'PSV' or 'WiiU'
        or 'NS' or 'PSN' or 'WS' or 'VC' or 'NG'
        or 'SCD' or 'PCE' or '3DO' or 'GG' or 'OSX'
        or 'XBL' or 'PCFX' or 'WW' or 'Series' or 'All'
        or 'iOS' or '5200' or 'And' or 'DSiW' or 'Lynx'
        or 'MS' or '7800' or 'ZXS' or 'ACPC' or 'DSi'
        or 'AJ' or 'WinP' or 'Mob' or 'Linux' or 'iQue'
        or 'Amig' or 'GIZ' or 'VB' or 'Ouya' or 'NGage'
        or 'XS' or 'Int' or 'CV' or 'Arc' or 'PS5'
        or 'OR' or 'CDi' or 'CD32' or 'BRW' or 'MSX'
        or 'MSD' or 'C64' or 'ApII' or 'AST') and (total_sales >= 0.6):
        return '1'
    else:
        return '0'


def delete_outlier(df):
    for column_name, column_type in df.dtypes.items():
        if column_type == 'float64':
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].index
            df = df.drop(outlier_indices)

    return df


def delete_unvalid_row(df, attr):
    details = pd.read_csv('info.csv')
    row_index = details[details['name'] == attr].index[0]
    range_attr = details['range'][row_index]
    range_type_attr = details['range_type'][row_index]
    i = 0

    if range_type_attr == 'range':

        range_attr_arr = range_attr.split(',')
        for datapoints in df[attr]:
            if not (float(range_attr_arr[0]) <= datapoints <= float(range_attr_arr[1])):
                df.drop(df.index[i], inplace=True)
                print('i:', i)

            else:

                i += 1


    elif range_type_attr == 'list':
        range_attr_arr = range_attr.split(',')
        for datapoints in df[attr]:
            if not (str(datapoints) in range_attr_arr):
                df.drop(df.index[i], inplace=True)
                print('i:', i)

            else:

                i += 1

    else:
        for datapoints in df[attr]:
            if not re.match(range_attr, datapoints):
                df.drop(df.index[i], inplace=True)
                print('i:', i)

            else:

                i += 1
    print("reduction row is done")
    return df


def create_total_sales_categorical(df, feature):
    df[feature + '_category'] = df[feature].apply(
        lambda x: 'low_sales' if float(x) < 6.77 else ('medium_sales' if float(x) <= 13.54 else 'high_sales'))
    df = df.drop(['total_sales'], axis=1)
    return df


def create_critic_score_categorical(df, feature):
    if feature not in df.columns:
        raise KeyError(f"'{feature}' column not found in the DataFrame.")
    df[feature + '_category'] = df[feature].apply(
        lambda x: 'low_score' if float(x) < 3.3 else ('medium_score' if float(x) <= 6.6 else 'high_score'))
    df = df.drop(['critic_score'], axis=1)
    return df


def text_analyze(df):
    for column_name, column_type in df.dtypes.items():
        if column_type == 'object':
            # Stemming
            stemmer = PorterStemmer()
            df[column_name] = df[column_name].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            df[column_name] = df[column_name].apply(
                lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    return df


def visualization(df):
    for column in df.columns:
        if column != 'title' and column != 'img':
            if df[column].dtype == 'object':  # For categorical data
                counts = df[column].value_counts()
                plt.figure(figsize=(8, 6))
                plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
                plt.title(f'Pie chart of {column}')
                plt.axis('equal')
                plt.show()
            else:  # For numerical data
                # You can modify this part to group numerical data into bins if needed
                plt.figure(figsize=(8, 6))
                plt.hist(df[column], bins=10, edgecolor='black')
                plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.show()


def compare(df):
    all_scores = df['User Score']
    action_scores = df[df['Genres'] == 'Action']['User Score']
    all_scores = all_scores.dropna()
    action_scores = action_scores.dropna()
    t_stat, p_value = ttest_ind(all_scores, action_scores)
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference between user scores of Action and all games.")
    else:
        print("There is no significant difference between user scores of Action and all games.")

def Extracting_Frequent_Pattern(df):
    # Define mapping for various combinations
    genres_console_list = list(zip(df['genre'].astype(str), df['console'].astype(str)))
    genres_publisher_list = list(zip(df['genre'].astype(str), df['publisher'].astype(str)))
    publisher_developer_console_list = list(
        zip(df['publisher'].astype(str), df['developer'].astype(str), df['console'].astype(str)))
    total_sales_genres_console_list = list(
        zip(df['total_sales_category'].astype(str), df['genre'].astype(str), df['console'].astype(str)))
    console_genres_publisher_developer_list = list(
        zip(df['console'].astype(str), df['genre'].astype(str), df['publisher'].astype(str),
            df['developer'].astype(str)))
    total_sales_developer_publisher_genre_list = list(
        zip(df['total_sales_category'].astype(str), df['developer'].astype(str), df['publisher'].astype(str),
            df['genre'].astype(str)))
    genres_total_sales_list = list(zip(df['genre'].astype(str), df['total_sales_category'].astype(str)))

    # Helper function to perform apriori analysis and save results
    def analyze_and_save(data_list, filename_prefix):
        te = TransactionEncoder()
        te_ary = te.fit(data_list).transform(data_list)
        new_df = pd.DataFrame(te_ary, columns=te.columns_)
        result = apriori(new_df, min_support=0.1, use_colnames=True)
        # rules = association_rules(result, metric="confidence", min_threshold=0.7)
        # print(rules)
        # Convert 'itemsets' column back to a string representation
        result['itemsets'] = result['itemsets'].apply(lambda x: ', '.join(map(str, x)))
        # Save the DataFrame to a CSV file
        new_df.to_csv(f"frequency_pattern\\{filename_prefix}_transaction.csv", index=False)
        result.to_csv(f"frequency_pattern\\{filename_prefix}_patterns.csv", index=False)
        return result

    # Analyze each attribute combination
    genres_console_result = analyze_and_save(genres_console_list, 'genres_console')
    genres_publisher_result = analyze_and_save(genres_publisher_list, 'genres_publisher')
    publisher_developer_console_result = analyze_and_save(publisher_developer_console_list,
                                                          'publisher_developer_console')
    total_sales_genres_console_result = analyze_and_save(total_sales_genres_console_list, 'total_sales_genres_console')
    console_genres_publisher_developer_result = analyze_and_save(console_genres_publisher_developer_list,
                                                                 'console_genres_publisher_developer')
    total_sales_developer_publisher_genre_result = analyze_and_save(total_sales_developer_publisher_genre_list,
                                                                    'total_sales_developer_publisher_genre')
    genres_total_sales_result = analyze_and_save(genres_total_sales_list, 'genres_total_sales')

    return (
        genres_console_result,
        genres_publisher_result,
        publisher_developer_console_result,
        total_sales_genres_console_result,
        console_genres_publisher_developer_result,
        total_sales_developer_publisher_genre_result,
        genres_total_sales_result
    )


def Clustering(df):
    group_0 = ['publisher_numeric', 'developer_numeric', 'console_numeric']
    df['publisher_numeric'] = convet_to_numerics(df, 'publisher', 'publisher_numeric')
    df['developer_numeric'] = convet_to_numerics(df, 'developer', 'developer_numeric')
    df['console_numeric'] = convet_to_numerics(df, 'console', 'console_numeric')
    df = perform_clustering(df, group_0, "group_0", K=2)
    # visualize_clusters(df, group_0, 'group_0', k=2)

    group_1 = ['genre_numeric', 'console_numeric']
    df['genre_numeric'] = convet_to_numerics(df, 'genre', 'genre_numeric')
    df['console_numeric'] = convet_to_numerics(df, 'console', 'console_numeric')
    df = perform_clustering(df, group_1, "group_1", K=2)
    # visualize_clusters(df, group_1, 'group_1', k=2)

    group_2 = ['publisher_numeric', 'developer_numeric']
    df['publisher_numeric'] = convet_to_numerics(df, 'publisher', 'publisher_numeric')
    df['developer_numeric'] = convet_to_numerics(df, 'developer', 'developer_numeric')
    df = perform_clustering(df, group_2, "group_2", K=2)
    # visualize_clusters(df, group_2, 'group_2', k=2)

    group_3 = ['publisher_numeric', 'developer_numeric', 'console_numeric', 'genre_numeric']
    df['publisher_numeric'] = convet_to_numerics(df, 'publisher', 'publisher_numeric')
    df['developer_numeric'] = convet_to_numerics(df, 'developer', 'developer_numeric')
    df['console_numeric'] = convet_to_numerics(df, 'console', 'console_numeric')
    df['genre_numeric'] = convet_to_numerics(df, 'genre', 'genre_numeric')
    df = perform_clustering(df, group_3, "group_3", K=2)
    # visualize_clusters(df, group_3, 'group_3', k=2)

    df.to_csv("clustring//clustring.csv", index=False)

    # Separate CSV for agglomerative clustering results
    df.to_csv("clustring//agglomerative.csv", index=False)


def perform_clustering(df, name, path, K=None):
    selected_columns = df[name].values
    # selected_column = df[[name]].values
    # selected_column = df[name]

    if K is not None:
        kmeans = KMeans(random_state=42, n_clusters=10)
        # kmeans = KMeans(random_state=42, n_clusters=K, n_init=1)
        kmeans.fit(selected_columns)
        centroids = kmeans.cluster_centers_
        df[f'{path}_Kmeans'] = kmeans.labels_
        labels = kmeans.labels_


        plt.figure(figsize=(8, 6))
        plt.scatter(selected_columns[:, 0], selected_columns[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200, label='Centroids')
        plt.title('K-Means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()


        agglomerative = AgglomerativeClustering(n_clusters=K)
        agglomerative.fit(selected_columns)
        df[f'{path}_agglomerative'] = agglomerative.labels_

    else:
        agglomerative = AgglomerativeClustering()
        agglomerative.fit(selected_columns)
        df[f'{path}_agglomerative_cluster'] = agglomerative.labels_

    inertias = []
    for i in range(1, 16):
        kmeans = KMeans(random_state=42, n_clusters=i, n_init=1)
        kmeans.fit(selected_columns)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 16), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(f'clustring//{path}.png')
    plt.close()

    return df

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
def visualize_clusters(data, features, cluster_column, k):
    all_features = ['console', 'genre', 'publisher', 'developer']
    new_data = data[all_features]

    # Create pie charts for each non-clustered feature in each cluster
    for feature in all_features:
        fig, axes = plt.subplots(1, k, figsize=(15, 8))
        for cluster_label in range(k):
            feature_values = data[data[f'{cluster_column}_Kmeans'] == cluster_label][feature].value_counts()
            axes[cluster_label].pie(feature_values, labels=feature_values.index, autopct='%1.1f%%')
            axes[cluster_label].set_title(f'Cluster {cluster_label + 1} - {feature} Distribution')

        directory = f'clustring//{cluster_column}'
        ensure_directory_exists(directory)
        plt.savefig(f'{directory}//{cluster_column}_kmeans_{cluster_label + 1}_{feature} Distribution.png')
        plt.close()

    for feature in all_features:
        fig, axes = plt.subplots(1, k, figsize=(15, 8))
        for cluster_label in range(k):
            feature_values = data[data[f'{cluster_column}_agglomerative'] == cluster_label][feature].value_counts()
            axes[cluster_label].pie(feature_values, labels=feature_values.index, autopct='%1.1f%%')
            axes[cluster_label].set_title(f'Cluster {cluster_label + 1} - {feature} Distribution')

        directory = f'clustring//{cluster_column}'
        ensure_directory_exists(directory)
        plt.savefig(f'{directory}//{cluster_column}__agglomerative_{cluster_label + 1}_{feature} Distribution.png')
        plt.close()


def convet_to_numerics(df,data_attribute , new_name):
    one_hot_encoded = pd.get_dummies(df[data_attribute], prefix=data_attribute)
    label_encoder = LabelEncoder()
    df[new_name] = label_encoder.fit_transform(df[data_attribute])
    return df[new_name]

def convert_to_numeri(series):
    codes, uniques = pd.factorize(series)
    return codes

# def convert_to_numeri(data_attribute):
#     range_value = np.unique(str(data_attribute))
#     num_value = len(range_value)
#
#     if num_value == len(data_attribute):  # this atrribute is id
#         return False, None, None
#
#     value_mapping = {value: index + 1 for index, value in enumerate(range_value)}
#     mapped_datas = data_attribute.replace(value_mapping)
#     return True, value_mapping, mapped_datas

def Classification(df):

    # Convert Release Date to year
    # df['Release Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year

    # Map Product Rating to numeric values
    # rating_map = {
    #     'Rated E For Everyone': 1,
    #     'Rated T For Teen': 2,
    #     'Rated M For Mature': 3,
    #     'Rated AO For Adults Only': 4,
    #     'Rated RP For Rating Pending': 0
    # }
    # df['Product Rating Score'] = df['Product Rating'].map(rating_map).fillna(-1)

    # Convert Genres to numeric features
    # df['Genres_NUM'] = convert_to_numeri(df['Genres Splitted'])

    # Convert Developer and Publisher to numeric features
    # df['Developer_NUM'] = convert_to_numeri(df['Developer'])
    # df['Publisher_NUM'] = convert_to_numeri(df['Publisher'])

    # Features to be used in the model
    # X_name = ['Release Year', 'User Ratings Count', 'Product Rating Score', 'Genres_NUM', 'Developer_NUM',
    #           'Publisher_NUM']


    f_name = ['Title_numeric', 'Developer_numeric', 'Publisher_numeric', 'Genres_numeric', 'Genres_Splitted_numeric', 'Product_Rating_numeric', 'User Score', 'User Ratings Count','Platforms_Info_numeric']

    df['Title_numeric'] = convet_to_numerics(df, 'Title', 'Title_numeric')
    df['Developer_numeric'] = convet_to_numerics(df, 'Developer', 'Developer_numeric')
    df['Publisher_numeric'] = convet_to_numerics(df, 'Publisher', 'Publisher_numeric')
    df['Genres_numeric'] = convet_to_numerics(df, 'Genres', 'Genres_numeric')
    df['Genres_Splitted_numeric'] = convet_to_numerics(df, 'Genres Splitted', 'Genres_Splitted_numeric')
    df['Product_Rating_numeric'] = convet_to_numerics(df, 'Product Rating', 'Product_Rating_numeric')
    df['Platforms_Info_numeric'] = convet_to_numerics(df, 'Platforms Info', 'Platforms_Info_numeric')

    data = df[f_name]

    #Correlation
    correlation_matrix = data.corr()
    print(correlation_matrix['User Score'].sort_values(ascending=False))

    #numeric_attribute
    # data = df.dropna()
    # data = data.fillna(0)
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    X = data[numeric_features].drop('User Score', axis=1)
    y = data['User Score']
    # F, pval = f_classif(X, y)
    # print(dict(zip(X.columns, pval)))

    #Mutual Information
    # mi = mutual_info_regression(X, y)
    # mi_series = pd.Series(mi, index=X.columns)
    # print(mi_series.sort_values(ascending=False))

    #Regression
    # model = LinearRegression()
    # model.fit(X, y)
    # importance = model.coef_
    # print(dict(zip(X.columns, importance)))

    #decision_tree
    # tree_model = DecisionTreeRegressor()
    # tree_model.fit(X, y)
    # importance = tree_model.feature_importances_
    # print(dict(zip(X.columns, importance)))

    #SVR
    # svr_model = SVR(kernel='linear')
    # svr_model.fit(X, y)
    # importance = svr_model.coef_
    # print(dict(zip(X.columns, importance.flatten())))


def main():
    # part1
    # df = pd.read_csv('vgchartz-2024.csv')
    # df = pd.read_csv('all_video_games(cleaned).csv')
    # df = Knowing_dataset(df)

    # part2
    # df = pd.read_csv('vgchartz-2024.csv')
    # consistency_ratio, num_inconsistent, inconsistent_rows = check_consistency(df)
    # print(f"Consistency Ratio: {consistency_ratio:.2f}")
    # print(f"Number of Inconsistent Rows: {num_inconsistent}")
    # print("Inconsistent Rows:")
    # print(inconsistent_rows)
    # Data_quality_assessment(df)
    # print('done')

    # part3
    # df = pd.read_csv('vgchartz-2024.csv')
    # # df = pd.read_csv('all_video_games(cleaned).csv')
    #
    # print('number of data before drop_duplicates:\n')
    # print(len(df['title']))
    # df = df.drop_duplicates(subset=['title'])
    # print('number of data after drop_duplicates:\n')
    # print(len(df['title']))
    #
    # df = handle_missing_data(df)
    #
    # normalization(df, ['total_sales', 'pal_sales', 'other_sales'])
    #
    # df['popular_genre'] = df.apply(lambda row: evaluate_genre(row['genre'], row['total_sales_normal']), axis=1)
    # df['popular_console'] = df.apply(lambda row: evaluate_console(row['console'], row['total_sales_normal']), axis=1)
    #
    # print('number of data before delete_outlier:\n')
    # print(len(df['title']))
    # df = delete_outlier(df)
    # print('number of data after delete_outlier:\n')
    # print(len(df['title']))
    #
    # df = delete_unvalid_row(df, 'total_sales')
    # df = delete_unvalid_row(df, 'other_sales')
    #
    # df = create_total_sales_categorical(df, 'total_sales')
    # # df = create_critic_score_categorical(df, 'critic_score')
    #
    # df = text_analyze(df)
    #
    # # visualization(df)
    #
    # df1 = pd.read_csv('all_video_games(cleaned).csv')
    # compare(df1)
    #
    # df.to_csv('phase3.csv', index=False)

    # part 4

    # df = pd.read_csv('vgchartz-2024.csv')
    # df1 = pd.read_csv('all_video_games(cleaned).csv')
    # df = combination_df(df, df1)
    # df = handle_missing_data(df)
    # df.to_csv('phase4.csv', index=False)

## phase_2

# part 1
    # df = pd.read_csv('phase3.csv')
    # df = Extracting_Frequent_Pattern(df)

# part 2
    df = pd.read_csv('phase3.csv')
    df = Clustering(df)

# part 3
#     df = pd.read_csv('all_video_games(cleaned).csv')
#     df = Classification(df)



if __name__ == "__main__":
    main()