from funct import funciones as fun
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import subprocess

# Título de la página
st.set_page_config(page_title = 'Calidad del Café', page_icon= ":coffee:")
st.title('Calidad del Café:')
st.header('Una Clasificación con Machine Learning')
st.image('https://sepuedeonosepuede.com/wp-content/uploads/2022/10/comer-granos-de-cafe-scaled.jpg',
         caption='Giacomo Salerno')
st.divider()

# Datasets a trabajar
arabica = pd.read_csv('data/raw/arabica_data_cleaned.csv')
arabicapro = pd.read_csv('data/processed/arabica_processed.csv')
train = pd.read_csv('data/train/arabica_train.csv')
test = pd.read_csv('data/test/arabica_test.csv')

# Sidebar
st.sidebar.title('Contenido')
st.sidebar.divider()
# ¿Agua Sucia o Café
if st.sidebar.button('¿Agua Sucia o Café?'):
    st.title('¡A nadie le gusta un mal café!')
    st.markdown('Es por eso que nuestra empresa busca siempre lo mejor para nuestros clientes, un café de aroma irresistible, con cuerpo balanceado, que no sea demasiado ácido, y sobre todo, que esté muy bien de precio.')
    st.markdown('La idea principal es crear un clasificador de calidad de café en tres distintas categorías: estándar, bueno y premium; para así asegurarnos de siempre comprar productos de calidad por el precio ideal.')
    st.markdown('Para ello, se utilizan diferentes variables: país de origen, variedad de café, procesado, año de cosecha, humedad, color, defectos e incluso la altura en que ha sido cultivado. Todo esto se introduce en un modelo predictivo de machine learning, para así saber qué tan bueno es el producto antes de realizar la compra.')

# Datos
if st.sidebar.button('Datos'):
    st.header('Datos en Crudo')
    arabica
    arabica.shape
    st.divider()
    st.header('Variables Categóricas')
    st.image('app/img/category.png')
    st.divider()
    st.header('Valores Nulos')
    st.image('app/img/missingvalues.png')
    st.divider()
    st.header('Correlaciones')
    st.image('app/img/totalcorr.png')
    st.divider()
    st.header('Distribución de Calidad')
    st.image('app/img/distcalidad.png')
    st.divider()
    st.header('Balanceo')
    st.image('app/img/unbalanced.png')

# Procesamiento 
if st.sidebar.button('Procesamiento'):
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Original', 'Limpieza', 'Agrupación', 'Correcciones', 'Etiquetado', 'Balanceado', 'Final'])

    with tab0:
        st.header('Datos en Crudo')
        arabica
        arabica.shape
        st.divider()
        

    with tab1:
        st.header('Limpieza de Datos')
        st.code("""
        # Columnas con las que me quiero quedar.
        arabica = arabica[['Country.of.Origin', 'Variety', 'Processing.Method', 'Moisture',
                        'Harvest.Year', 'Color','unit_of_measurement', 'altitude_mean_meters',
                        'Category.One.Defects', 'Category.Two.Defects','Total.Cup.Points']]

        # Drop missing values.
        arabica = arabica.dropna()
        """)
        st.divider()
        st.subheader('Resultado')
        arabica = fun.clean_data(arabica)
        arabica
        arabica.shape
        st.divider()

    with tab2:
        st.header('Agrupación de Minorías')
        st.code("""
        # Agrupar valores minoritarios.
        otros = arabica['Country.of.Origin'].value_counts() <= 5
        for i in range(len(otros.index)):
            if(otros[i]):
                arabica.loc[arabica["Country.of.Origin"] == otros.index[i], "Country.of.Origin"] = "Other"

        otros = arabica['Variety'].value_counts() == 1
        for i in range(len(otros.index)):
            if(otros[i]):
                arabica.loc[arabica["Variety"] == otros.index[i], "Variety"] = "Other"
        
        color = {'Green': 'None',
                 'Blue' : ['Blue-Green', 'Bluish-Green']}
        arabica['Color'] = arabica['Color'].map(lambda x: next((k for k, v in color.items() if x in v), x))
        """)
        st.divider()
        st.header('Resultado')

        column1, column2 = st.columns(2)

        with column1:
            st.subheader('Antes')
            arabica['Country.of.Origin']
            unico = arabica['Country.of.Origin'].nunique()
            unico
            st.divider()
            arabica['Variety']
            unico = arabica['Variety'].nunique()
            unico
            st.divider()
            arabica['Color']
            unico = arabica['Color'].nunique()
            unico
        
        with column2:
            st.subheader('Después')
            arabica = fun.group(arabica)
            arabica['Country.of.Origin']
            unico = arabica['Country.of.Origin'].nunique()
            unico
            st.divider()
            arabica['Variety']
            unico = arabica['Variety'].nunique()
            unico
            st.divider()
            arabica['Color']
            unico = arabica['Color'].nunique()
            unico

    with tab3:
        st.header('Otras Correcciones')
        st.code("""
        # Corregir año de cosecha.
        year = {'2015/2016' : 2016,
                '2013/2014' : 2014,
                '2017 / 2018' : 2018,
                '2014/2015' : 2015,
                '2011/2012' : 2012,
                '2016 / 2017' : 2017}
        arabica['Harvest.Year'] = arabica['Harvest.Year'].replace(year).astype(int)

        # Convertir altitudes a mismas unidades.
        mask = arabica['unit_of_measurement'].eq('ft')
        arabica.loc[mask, ['altitude_mean_meters']] /= 3.281
        arabica= arabica.drop(columns= "unit_of_measurement")

        # Eliminar outliers irreales en altitud
        arabica= arabica[~(arabica["altitude_mean_meters"] > 9000)]
        """)
        st.divider()
        st.header('Resultado')

        column1, column2 = st.columns(2)

        with column1:
            st.subheader('Antes')
            valor = arabica['Harvest.Year'].unique()
            st.markdown('Valores únicos de [Harvest.Year]:')
            st.markdown(valor)
            st.divider()
            valor = arabica['unit_of_measurement'].unique()
            st.markdown('Unidades de altitud:')
            st.markdown(valor)
            st.image('app/img/outlierspre.png')

        with column2:
            st.subheader('Después')
            arabica = fun.correct(arabica)
            valor = arabica['Harvest.Year'].unique()
            st.markdown('Valores únicos de [Harvest.Year]:')
            st.markdown(valor)
            st.write("")
            st.write("")
            st.write("")
            st.divider()
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.image('app/img/outlierspost.png')
    
    with tab4:
        st.header('Etiquetado de Categorías')
        st.code("""
        # Binarizar los colores.
        lb = LabelBinarizer()

        arabica.Color = lb.fit_transform(arabica.Color)

        # Encoding de otras variables categóricas.
        le = LabelEncoder()

        columnas = ["Country.of.Origin", "Variety", "Processing.Method"]

        for columna in columnas:
            arabica[columna] = le.fit_transform(arabica[columna])

        # Defino la columna target.
        calidad = [0, 1, 2]
        calif = [0, 80, 85, 100]
        arabica['Calidad'] = pd.cut(arabica['Total.Cup.Points'], bins=calif, labels=calidad)
        """)
        st.divider()
        st.header('Resultados')

        column1, column2 = st.columns(2)

        with column1:
            st.subheader('Antes')
            arabica['Color']
            st.divider()
            st.dataframe(arabica[['Country.of.Origin', 'Variety', 'Processing.Method']])

        with column2:
            st.subheader('Después')
            arabica = fun.label(arabica)
            arabica['Color']
            st.divider()
            st.dataframe(arabica[['Country.of.Origin', 'Variety', 'Processing.Method']])

    with tab5:
        st.header('Balanceado de Datos')
        st.code("""
        # Balanceo el dataframe.
        ros = RandomOverSampler(random_state=5)
        X_resampled, y_resampled = ros.fit_resample(arabica.loc[:, 'Country.of.Origin':'Total.Cup.Points'], arabica['Calidad'])
        df_resampled = pd.DataFrame(X_resampled, columns=arabica.loc[:, 'Country.of.Origin':'Total.Cup.Points'].columns)
        df_resampled['Calidad'] = y_resampled
        df_balanced = pd.concat([arabica, df_resampled], ignore_index=True)

        # Quito la "chuleta"
        df_balanced.drop(columns='Total.Cup.Points', inplace=True)
        """)
        st.divider()
        st.header('Resultados')
        column1, column2 = st.columns(2)
        
        with column1:
            st.subheader('Antes')
            arabica
            arabica.shape
            st.image('app/img/unbalanced.png')

        with column2:
            st.subheader('Después')
            arabica = fun.balance(arabica)
            arabica
            arabica.shape
            st.image('app/img/balanced.png')
    
    with tab6:
        st.header('Dataframe Procesado')
        arabicapro
        arabicapro.shape
        st.divider()
        st.header('Train')
        train
        train.shape
        st.header('Test')
        test
        test.shape
        st.balloons()

# Machine Learning

if st.sidebar.button('Modelos'):
    pretab, tab0, tab1, tab2, tab3 = st.tabs(['Primeros Modelos', 'Hiperparametrización', 'Modelo Final', 'Entrenamiento', 'Validación'])

    with pretab:
        st.header('Regresión Logística')
        st.code('''
        logistic_regression = LogisticRegression()

        logistic_params = {'penalty': ['l1', 'l2'],
                        'C': [0.1, 1.0, 10.0],
                        'solver': ['liblinear', 'saga'],
                        'max_iter': [100, 500, 1000]}

        logistic_gs = GridSearchCV(estimator=logistic_regression, param_grid=logistic_params, cv=5, scoring='accuracy', n_jobs=-1)
        logistic_gs.fit(Xtrain, ytrain)
        pred = logistic_gs.predict(Xtest)
        ''')
        st.markdown('* Accuracy: 56,99 %')
        st.markdown('* Precision: [41,45 % ; 79,78 % ; 29,55 %]')
        st.markdown('* Recall: [58,39 % ; 55,55 % ; 63,15 %]')
        st.divider()
        st.header('Ada Boost + Árbol de Clasificación')
        st.code('''
        base_classifier = DecisionTreeClassifier()

        ada_boost = AdaBoostClassifier(base_estimator=base_classifier)

        ada_params = {'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.5, 1.0],
                    'base_estimator__max_features': [2, 3, 4],
                    'base_estimator__max_depth': [3, 5, 7],
                    'base_estimator__min_samples_leaf': [2, 4, 6],
                    'base_estimator__max_leaf_nodes': [6, 7, 8]}

        ada_gs = GridSearchCV(estimator=ada_boost, param_grid=ada_params, cv=5, scoring='accuracy', n_jobs=-1)
        ada_gs.fit(Xtrain, ytrain)
        pred = ada_gs.predict(Xtest)
        ''')
        st.markdown('* Accuracy: 78,36 %')
        st.markdown('* Precision: [66,32 % ; 86,17 % ; 75,36 %]')
        st.markdown('* Recall: [76,19 % ; 73,80 % ; 92,72 %]')
    
    with tab0:
        st.header('Pipeline')
        st.code("""
        pipe = Pipeline(steps = [("scaler", StandardScaler()),
                       ("selectkbest", SelectKBest()),
                       ("pca", PCA()),
                       ('classifier', RandomForestClassifier())])

        logistic_params = {'selectkbest__k' : np.arange(3,9),
                        'pca__n_components': [7, 8, 9],
                        'classifier': [LogisticRegression(solver='liblinear')],
                        'classifier__penalty': ['l1','l2']}

        rf_params = {'scaler' : [StandardScaler(), None],
                    'selectkbest__k' : np.arange(3,9),
                    'pca__n_components': [7, 8, 9],
                    'classifier': [RandomForestClassifier()],
                    'classifier__max_features': [2, 3, 4],
                    'classifier__max_depth': [3, 5, 7]}

        gb_params = {'scaler' : [StandardScaler(), None],
                    'selectkbest__k' : np.arange(3,9),
                    'pca__n_components': [7, 8, 9],
                    'classifier': [GradientBoostingClassifier()],
                    'classifier__max_features': [2, 3, 4],
                    'classifier__max_depth': [6, 7, 8]}

        knn_params = {'selectkbest__k' : np.arange(3,9),
                    'pca__n_components': [7, 8, 9],
                    'classifier': [KNeighborsClassifier()],
                    'classifier__n_neighbors': [5, 7, 12]}

        svm_params = {'selectkbest__k' : np.arange(3,9),
                    'pca__n_components': [7, 8, 9],
                    'classifier': [SVC()],
                    'classifier__C': [0.1, 1, 10]}

        search_space = [logistic_params, rf_params, gb_params, knn_params, svm_params]

        clf = GridSearchCV(estimator = pipe,
                            param_grid = search_space,
                            cv = 3,
                            scoring = "accuracy",
                            n_jobs = -1)
        """)
        st.markdown('El mejor clasificador en este caso ha sido el Gradient Boosting Classifier.')
        st.markdown('Como mejores parámetros, el Grid Search ha devuelto:')
        st.markdown('* max_depth = 7')
        st.markdown('* max_features = 2')
        st.markdown('* PCA: 9')
        st.markdown('* scaler: None')
        st.markdown('* K = 9')
        st.divider()
        st.header('Entrenamiento')
        st.subheader('Datos Utilizados')
        arabicapro
        arabicapro.shape
        column1, column2 = st.columns([2, 1])
        
        with column1:
            st.subheader('X')
            X = train.drop(columns='Calidad')
            X
            X.shape

        with column2:
            st.subheader('y')
            y = train.Calidad
            y
            y.shape

        st.divider()
        st.header('Datos de Evaluación')
        X = test.drop(columns='Calidad')
        X
        X.shape
        st.header('Resultados')
        st.subheader('Métricas')
        st.markdown('Las métricas utilizadas para la evaluación han sido accuracy, precision y recall.')
        st.markdown('* Accuracy: 97,40 %')
        st.markdown('* Precision: [100 % ; 94,68 % ; 100 %]')
        st.markdown('* Recall: [94,14 % ; 100 % ; 96,21 %]')
        st.subheader('Matriz de Confusión')
        st.image('app/img/trainmatrix.png')

    with tab1:
        st.header('Gradient Boosting Classifier')
        st.code('''
        best_params = {'selectkbest__k': 9,
                       'pca__n_components': 9,
                       'classifier': GradientBoostingClassifier(max_depth = 7,
                                                                max_features = 2)}

        best_model = Pipeline(steps = [("selectkbest", SelectKBest( k = best_params['selectkbest__k'])),
                                       ("pca", PCA(n_components = best_params['pca__n_components'])),
                                       ("classifier", best_params['classifier'])])
        ''')
        st.markdown('El modelo final se trata de un Gradient Boosting Classifier, acoplado a un Pipeline para también aplicar modelos no supervisados como el Principal Component Analysis (PCA) y otros procesos como la selección de mejores features.')
    with tab2:
        st.header('Datos de Entrenamiento')
        train
        train.shape

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader('X')
            X = train.drop(columns='Calidad')
            X
            X.shape

        with col2:
            st.subheader('y')
            y = train.Calidad
            y
            y.shape
        
        st.divider()
        st.header('Guardado del Modelo Entrenado')
        st.code('''
        with open(yaml['output_file'], 'wb') as file:
            pickle.dump(best_model, file)
        ''')
        st.markdown('En lugar de la ruta, se utiliza el parámetro preestablecido en el archivo .yaml')
    
    with tab3:
        st.header('Evaluación del Modelo')
        st.subheader('Cargado del Modelo')
        st.code('''
        with open('models/modelo_final.pkl', 'rb') as file:
            modelo = pickle.load(file)
        ''')
        st.divider()
        st.header('Datos de Evaluación')
        test
        test.shape
        st.divider()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader('X')
            X = test.drop(columns='Calidad')
            X
            X.shape

        with col2:
            st.subheader('y')
            y = test.Calidad
            y
            y.shape
        st.divider()
        st.header('Resultados')
        st.subheader('Métricas')
        st.markdown('* Accuracy: 97,89 %')
        st.markdown('* Precision: [100 % ; 95,63 % ; 100 %]')
        st.markdown('* Recall: [94,51 % ; 100 % ; 97,61 %]')
        st.subheader('Matriz de Confusión')
        st.image('app/img/evalmatrix.png')
        st.balloons()

if st.sidebar.button('Conclusiones'):
    st.title('Conclusiones')
    st.markdown('El modelo predictivo parece tener buenos resultados ante los datos aportados, teniendo una precisión casi perfecta.')
    st.markdown('El objetivo principal era no confundir calidades "extrapoladas", es decir, predecir como prémium una muestra de calidad estándar o vicecersa; por lo que se puede decir que el objetivo ha sido alcanzado.')
    st.markdown('Las principales variables que más fuerza han tenido para alcanzar una clasificación correcta han sido los defectos de los frutos del café; si éste ha sido dañado por aves o insectos, o si la muestra en general tiene piedras o ramas, algo que se produce por la cosecha industrializada del producto (las muestras escogidas a mano poseen una mejor calidad, y por ende, están tasadas a un precio mayor).')

if st.sidebar.button('Demo'):
    subprocess.run(["streamlit", "run", "app/demo.py"])