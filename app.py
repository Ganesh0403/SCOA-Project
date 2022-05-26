import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import base64
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix
import statistics
import altair as alt
import random

class GeneticSelector():
    def __init__(self, estimator, n_gen, size, n_best, n_rand,
                 n_children, mutation_rate):
        # Estimator
        self.estimator = estimator
        # Number of generations
        self.n_gen = n_gen
        # Number of chromosomes in population
        self.size = size
        # Number of best chromosomes to select
        self.n_best = n_best
        # Number of random chromosomes to select
        self.n_rand = n_rand
        # Number of children created during crossover
        self.n_children = n_children
        # Probablity of chromosome mutation
        self.mutation_rate = mutation_rate

        if int((self.n_best + self.n_rand) / 2) * self.n_children != self.size:
            raise ValueError("The population size is not stable.")

    def initilize(self):
        population = []
        for i in range(self.size):
            chromosome = np.ones(self.n_features, dtype=np.bool)
            mask = np.random.rand(len(chromosome)) < 0.3
            chromosome[mask] = False
            population.append(chromosome)
        return population

    def fitness(self, population):
        X, y = self.dataset
        scores = []
        for chromosome in population:
            score = -1.0 * np.mean(cross_val_score(self.estimator, X[:, chromosome], y,
                                                   cv=5,
                                                   scoring="neg_mean_squared_error"))
            scores.append(score)
        scores, population = np.array(scores), np.array(population)
        inds = np.argsort(scores)
        return list(scores[inds]), list(population[inds, :])

    def select(self, population_sorted):
        population_next = []
        for i in range(self.n_best):
            population_next.append(population_sorted[i])
        for i in range(self.n_rand):
            population_next.append(random.choice(population_sorted))
        random.shuffle(population_next)
        return population_next

    def crossover(self, population):
        population_next = []
        for i in range(int(len(population)/2)):
            for j in range(self.n_children):
                chromosome1, chromosome2 = population[i], population[len(
                    population)-1-i]
                child = chromosome1
                mask = np.random.rand(len(child)) > 0.5
                child[mask] = chromosome2[mask]
                population_next.append(child)
        return population_next

    def mutate(self, population):
        population_next = []
        for i in range(len(population)):
            chromosome = population[i]
            if random.random() < self.mutation_rate:
                mask = np.random.rand(len(chromosome)) < 0.05
                chromosome[mask] = False
            population_next.append(chromosome)
        return population_next

    def generate(self, population):
        # Selection, crossover and mutation
        scores_sorted, population_sorted = self.fitness(population)
        population = self.select(population_sorted)
        population = self.crossover(population)
        population = self.mutate(population)
        # History
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))

        return population

    def fit(self, X, y):

        self.chromosomes_best = []
        self.scores_best, self.scores_avg = [], []

        self.dataset = X, y
        self.n_features = X.shape[1]

        population = self.initilize()
        for i in range(self.n_gen):
            population = self.generate(population)
            st.write("In generation", i, " the best loss is ",
                  self.scores_best[-1], "The average loss is ", self.scores_avg[-1])

        return self

    @property
    def support_(self):
        return self.chromosomes_best[-1]

    def plot_scores(self):
        st.write(plt.plot(self.scores_best, label='Best'))
        st.write(plt.plot(self.scores_avg, label='Average'))
        st.write(plt.legend())
        st.write(plt.ylabel('Loss'))
        st.write(plt.xlabel('Generation'))
        st.pyplot()

st.title('Feature Selection using Genetic Algorithm')
st.sidebar.title('Feature Selection using Genetic Algorithm')

st.markdown(
    'This application is used for feature selection using genetic algorithm on boston dataset')
st.sidebar.markdown(
    'This application is used for feature selection using genetic algorithm on boston dataset')

@st.cache(persist=True)
def load():
    dataset = load_boston()
    X, y = dataset.data, dataset.target
    features = dataset.feature_names
    return X, y, features

X,y,features = load()

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston["medv"] = boston_dataset.target

st.write('Shape of Dataset', X.shape)
# st.write('Number of classes', len(np.unique(y)))

def add_parameter_ui():
    params = dict()
    K = st.sidebar.slider("Select Generation", 1, 25)
    params["gen"] = K

    K = st.sidebar.slider("For Mapping", 1, 25)
    params["mp"] = K
    return params

params = add_parameter_ui()

def get_classifier(params):
    
    sel = GeneticSelector(estimator=LinearRegression(),
                      n_gen=params["gen"], size=200, n_best=40, n_rand=40,
                      n_children=5, mutation_rate=0.05)
    return sel


sel = get_classifier(params)

if st.sidebar.checkbox('Run  Algorithm', False):
    est = LinearRegression()
    score = -1.0 * cross_val_score(est, X, y, cv=5,
                                scoring="neg_mean_squared_error")
    st.write("CV MSE before feature selection: {:.4f}".format(np.mean(score)))
    sel.fit(X,y)
    sel.plot_scores()
    score = -1.0 * \
    cross_val_score(est, X[:, sel.support_], y, cv=5,
                    scoring="neg_mean_squared_error")
    st.write("CV MSE after feature selection: {:.4f}".format(np.mean(score)))
    st.write(list(zip(features, sel.chromosomes_best[params["mp"]])))
else:
    if st.checkbox("ANALYZE DATASET"):
        if st.sidebar.checkbox("Preview Dataset"):
            if st.sidebar.button("Head"):
                st.write(boston.head())
            elif st.sidebar.button("Tail"):
                st.write(boston.tail())
            else:
                number = st.sidebar.slider("Select No of Rows", 1, boston.shape[0])
                st.write(boston.head(number))

        # show column names
        if st.checkbox("Show Column Names"):
            st.write(boston.columns)

        # show dimensions
        if st.checkbox("Show Dimensions"):
            st.write(boston.shape)

        # show summary
        if st.checkbox("Show Summary"):
            st.write(boston.describe())

        # show missing values
        if st.checkbox("Show Missing Values"):
            st.write(boston.isna().sum())

        # Select a column to treat missing values
        col_option = st.selectbox(
            "Select Column to treat missing values", boston.columns)

        # Specify options to treat missing values
        missing_values_clear = st.selectbox("Select Missing values treatment method", (
            "Replace with Mean", "Replace with Median", "Replace with Mode"))

        if missing_values_clear == "Replace with Mean":
            replaced_value = boston[col_option].mean()
            st.write("Mean value of column is :", replaced_value)
        elif missing_values_clear == "Replace with Median":
            replaced_value = boston[col_option].median()
            st.write("Median value of column is :", replaced_value)
        elif missing_values_clear == "Replace with Mode":
            replaced_value = boston[col_option].mode()
            st.write("Mode value of column is :", replaced_value)

        Replace = st.selectbox("Replace values of column?", ("Yes", "No"))
        if Replace == "Yes":
            boston[col_option] = boston[col_option].fillna(replaced_value)
            st.write("Null values replaced")
        elif Replace == "No":
            st.write("No changes made")

        # To change datatype of a column in a dataframe
        # display datatypes of all columns
        if st.checkbox("Show datatypes of the columns"):
            st.write(boston.dtypes.astype(str))

        # visualization
        # scatter plot
        col1 = st.selectbox('Which feature on x?', boston.columns)
        col2 = st.selectbox('Which feature on y?', boston.columns)
        fig = px.scatter(boston, x=col1, y=col2)
        st.plotly_chart(fig)

        # correlartion plots
        if st.checkbox("Show Correlation plots with Seaborn"):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(sns.heatmap(boston.corr()))
            st.pyplot()

    if st.sidebar.checkbox('Show Raw Data', False):
        st.write(boston)