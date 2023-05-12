            # Import packages
import streamlit as st

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

import dtreeviz

from lime.lime_tabular import LimeTabularExplainer

import svgwrite
import base64 
import io

            # Import dataset and create the models 


# Import the two datasets (now balanced datasets)
creditriskmodel = pd.read_csv('creditrisk01_ub.csv', index_col = False)  # For training and testing the models
creditriskclient = pd.read_csv('creditrisk02_b.csv', index_col = False) # Special clients for the test

# Create a list of the features
features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
            'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Define the features and target variable (loan_status)
X = creditriskmodel[features]
y = creditriskmodel['loan_status']

# Split the data into training- and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the decision tree on the training data
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)

# Fit the random forest on the training data
rfc = RandomForestClassifier(n_estimators=8, random_state=42)
rfc.fit(X_train, y_train)

# Fit the logistic regression on the training data
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)


            # Make the webapplication 


# Skip the warning: because of matplotlib import
st.set_option('deprecation.showPyplotGlobalUse', False)

# Give a title to the webapplication
st.markdown("<h1 style='text-align: center; color: #0077c2;'>Explainable AI voor consumenten kredietrisico</h1>", unsafe_allow_html=True)
# Introductory texts
st.write('Welkom bij de webapplicatie voor mijn onderzoek naar de voorkeuren van manieren van uitleg krijgen, vanuit kredietbeoordelaars. De toegepaste dataset is een openbare dataset die is gesimuleerd om een kredietrisicobeoordeling weer te geven. De dataset bevat verschillende factoren. Deze omvatten bijvoorbeeld leeftijd, inkomen en lengte van het dienstverband. Op basis van deze factoren wordt voorspeld of iemand een lening krijgt ja of nee. Het doel van dit experiment is na te gaan waar de voorkeuren voor het verkrijgen van uitleg vanuit AI-modellen liggen bij kredietbeoordelaars.')
st.write('Doordat er gebruik wordt gemaakt van een Amerikaanse dataset nog een aanmerking. In Amerika kennen kredietbeoordelingsbureaus zoals Experian, Equifax en TransUnion leningsgraden toe aan personen en bedrijven op basis van hun kredietwaardigheid. De leengraad is een score die aangeeft hoe waarschijnlijk het is dat een individu zijn schulden op tijd aflost. De rating is gewoonlijk gebaseerd op een aantal factoren, waaronder de kredietgeschiedenis van de persoon, het inkomen en andere financiële informatie. De rating wordt vaak uitgedrukt in een lettercijfer, gaande van "A" tot "F", waarbij "A" de hoogste rating is en "F" de laagste.')

            # Create dictionaries for overview 


# Create a dictionary to show the customers with the models
models = {
        'Klant 1': tree_clf,
        'Klant 2': tree_clf,
        'Klant 3': tree_clf,
        'Klant 4': tree_clf,
        'Klant 5': rfc,
        'Klant 6': rfc,
        'Klant 7': rfc,
        'Klant 8': rfc,
        'Klant 9': log_reg,
        'Klant 10': log_reg,
        'Klant 11': log_reg,
        'Klant 12': log_reg
    }

# Create a translation dictionary for the features
translations = {
    'person_age': 'Leeftijd (in jaar)',
    'person_income': 'Inkomen (in Dollar)',
    'person_emp_length': 'Lengte dienstverband (in jaar)',
    'loan_amnt': 'Bedrag lening (in Dollar)',
    'loan_int_rate': 'Renteprecentage',
    'loan_percent_income': 'Percentage inkomen',
    'cb_person_cred_hist_length': 'Lengte kredietgeschiedenis',
    'person_home_ownership': 'Bezit woning',
    'loan_intent': 'Intentie lening',
    'loan_grade': 'Leningsgraad',
    'cb_person_default_on_file': 'Historische wantbetaling'
}

# Create a translation library for decision tree and random forest indexes
translation_idx = {
    0: 'Leeftijd (in jaar)',
    1: 'Inkomen (in Dollar)',
    2: 'Lengte dienstverband (in jaar)',
    3: 'Bedrag lening (in Dollar)',
    4: 'Renteprecentage',
    5: 'Percentage inkomen',
    6: 'Lengte kredietgeschiedenis',
    7: 'Bezit woning',
    8: 'Intentie lening',
    9: 'Leningsgraad',
    10: 'Historische wantbetaling'
}

# Create conversion dictionaries for the features with 'integers' that were 'strings'
person_home_ownership01 = {0: 'Hypotheek', 1: 'Anders', 2: 'Eigen_huis', 3: 'Huur'}
loan_intent01 = {0: 'Schuldconsolidatie', 1: 'Onderwijs', 2: 'Woningverbetering', 3: 'Medisch', 4: 'Persoonlijk', 5: 'Investering'}
loan_grade01 = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
cb_person_default_on_file01 = {0: 'Nee', 1: 'Ja'}


            # Create the input for the webapplication


# Add a header for model selection
st.markdown("<h2 style='color: #0077c2;'>Bekijk de aanvragen</h2>", unsafe_allow_html=True)

# Create a menu to choose a client
selected_client = st.selectbox('Kies een klant:', options=creditriskclient['Client'].unique())

# Filter the dataframe to only show rows for the selected client from the above
filtered_clients = creditriskclient[creditriskclient['Client'] == selected_client]

# Get the loan status for the selected client
loan_status = filtered_clients['loan_status'].iloc[0]

# Show the approval status message in a block
if loan_status == 1:
    st.success(f"{selected_client} is goedgekeurd voor de leningaanvraag.")
else:
    st.error(f"{selected_client} is afgekeurd voor de leningaanvraag.")
    

             # Create functions for XAI methods 

# Define a function to show a text explanation 
def text_explanation_lime(model, client_data):
    # Rename the columns with the translation library
    X_renamed = X.rename(columns=translations)
    # Get the selected clients model and loan_status
    model = models[selected_client]
    loan_status = filtered_clients['loan_status'].iloc[0]
    # Make the text explainer using Lime
    exp = explainer.explain_instance(client_data.values.flatten(), model.predict_proba, num_features=len(X_renamed.columns))
    # Get the top three most important features to the prediction
    sorted_features = exp.as_list()
    # Delete the numbers and signs using RegEx
    top_features = [re.sub(r'\W+|\d+', ' ', translations.get(feature[0], feature[0])).strip() for feature in sorted_features[:3]]
    # Create a text explanation based on the top three features
    explanation_Lime = f'De voorspelling van het model is dat de leningaanvraag {"goedgekeurd" if loan_status == 1 else "afgewezen"} zal worden. '
    explanation_Lime += f'De belangrijkste factoren die hieraan bijdragen zijn: {", ".join(top_features)}.'
    # Show the explanation
    return explanation_Lime

# Define a function to show a local explanation in a figure 
def explain_client_lime(client_data, model):
    # Get the selected client's model
    model = models[selected_client]
    # Rename the columns with the translation library
    X_train_renamed = X_train.rename(columns=translations)
    # Make the Lime_explainer
    explainer_lime = LimeTabularExplainer(X_train_renamed.values,
                                                       feature_names=X_train_renamed.columns.tolist(),
                                                       class_names=['rejected', 'approved'],
                                                       mode='classification',
                                                       discretize_continuous=True)
    exp_lime = explainer_lime.explain_instance(client_data, model.predict_proba, num_features=6)
    # Use the renamed feature_names in the output
    exp_lime_list = [(translations.get(feature[0], feature[0]), feature[1]) for feature in exp_lime.as_list()]
    return exp_lime_list

# Define a function to show a decision tree
def show_decision_tree():
    # Drop the model column, otherwise it would not create the right path. Also get the right client
    x = creditriskclient.drop(columns=['model']).loc[filtered_clients.index.values[0]]
    # Create the figure
    viz = dtreeviz.model(tree_clf, creditriskmodel.drop(columns=['loan_status']), creditriskmodel['loan_status'], 
                     target_name='Leenstatus', feature_names=translation_idx, class_names=['Afgewezen', 'Toegewezen'])
    v = viz.view(x=x, fancy=False) # Histograms are not shown because of fancy. x=x is creating the path
    v.save('Images/decision.tree.svg') 
    # Show the updated SVG image in streamlit
    with open('Images/decision.tree.svg', 'r') as f:
        svg = f.read()
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = f'<img src="data:image/svg+xml;base64,{b64}" width="150%" height="auto"/>' # Width optimal at 150
        st.write(html, unsafe_allow_html=True) # Show as html, svg is ugly

# Define a function to get the most important tree out of the random forest
def get_most_important_tree(rfc, X, y):
    # Get feature importances for all trees
    importances = np.zeros((X.shape[1], len(rfc.estimators_)))
    for i, tree in enumerate(rfc.estimators_):
        importances[:, i] = tree.feature_importances_

    # Sum feature importances across all trees
    overall_importances = np.sum(importances, axis=1)

    # Select the tree with the highest overall feature importance
    most_important_tree_index = np.argmax(overall_importances)
    return rfc.estimators_[most_important_tree_index]

# Define a function to show a random forest
def show_random_forest():
    # Get the most important tree
    most_important_tree = get_most_important_tree(rfc, X, y)
    # Drop the model column, otherwise it would not create the right path. Also get the right client
    x = creditriskclient.drop(columns=['model']).loc[filtered_clients.index.values[0]]
    # Create the figure
    viz = dtreeviz.model(most_important_tree, creditriskmodel.drop(columns=['loan_status']), creditriskmodel['loan_status'], 
                     target_name='Leenstatus', feature_names=translation_idx, class_names=['Afgewezen', 'Toegewezen'])
    v = viz.view(x=x, fancy=False, show_just_path=True) # Histograms are not shown because of fancy. x=x is creating the path     
    v.save('Images/random.forest.svg') 
    # Show the updated SVG image in streamlit Images
    with open('Images/random.forest.svg', 'r') as f:
        svg = f.read()
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = f'<img src="data:image/svg+xml;base64,{b64}" width="100%" height="auto"/>' # Width optimal at 100
        st.write(html, unsafe_allow_html=True) # Show as html, svg is ugly

# Define a function to display the correlation figure
def display_correlation_figure(selected_client):
    # Get the selected client's model
    model = models[selected_client]
    # Get the X and y data for the creditrisk model
    X = creditriskmodel[features]
    y = creditriskmodel['loan_status']
    # Make predictions on the X data using the chosen model
    y_pred = model.predict(X)
    # Add the predicted loan_status to the X data
    X['loan_status'] = y_pred
    # Calculate the correlation table for the X data
    correlation_data = X.corr()
    # Rename the columns with the translation dictionary
    correlation_data = correlation_data.rename(columns=translations)
    # Rename the rows with the translation dictionary
    correlation_data = correlation_data.rename(index=translations)
    # Show the correlation table as a heatmap using a color scale with seaborn
    ax = sns.heatmap(correlation_data, vmin=-1, vmax=1, center=0, cmap='RdYlBu')
    ax.set_title("Correlaties tussen de factoren")
    st.pyplot()

# Define a function to calculate the recall
def recall(model, X_test, y_test):
    # Get the selected client's model
    model = models[selected_client]
    # Calculate the recall
    y_pred = model.predict(X_test[features])
    recall = recall_score(y_test, y_pred)
    return recall

# Define a function to calculate certainly (y_proba)
def certainty(model, X_test):
    # Get the selected client's model
    model = models[selected_client]
    # Calculate the certainly
    y_pred_proba = model.predict_proba(X_test)
    return np.mean(np.max(y_pred_proba, axis=1))

# Define a function to show te scores in words
def score_to_category(score):
    if score < 0.5:
        return "slecht"
    elif score < 0.8:
        return "goed"
    else:
        return "erg goed"
    
            # Create the buttons for the webapplication

# Add a header + text for showing the different XAI forms
st.markdown("<h2 style='color: #0077c2;'>Deel 1: Gebruik van Explainable AI methoden</h2>", unsafe_allow_html=True)
st.write('Onderstaand worden de eerste paar explainable AI methoden getoond. De XAI-methoden zijn klantspecifiek. Druk op een knop om de methode in beeld te brengen')

    # Working with XAI-methods part 1

# Create a button to show the information about the selected client  
if st.button("Methode 1"):
    # Show the loan_status of the chosen client
    loan_status = filtered_clients['loan_status'].iloc[0]
    probability = filtered_clients['probability'].iloc[0]
    # Determine the approval text based on the loan status
    approval_text = "goedgekeurd" if loan_status == 1 else "afgekeurd"
    # Modify the approval text based on the probability value
    if probability > 0.99:
        approval_text += ", op basis van de gegevens valt de klant onder een laag risico"
    elif probability >= 0.2:
        approval_text += ", op basis van de gegevens valt de klant onder een gemiddeld risico"
    else:
        approval_text += ", op basis van de gegevens valt de klant onder een hoog risico"
    # Show the approval text and client information
    st.write(f"Deze klant is {approval_text}. Hier zijn de gegevens over de desbetreffende klant:")
    # Rename columns with translations_features dataframe
    translated_features = filtered_clients[features].rename(columns=translations)
    # Convert selected features to their original string values, see conversation libraries 
    translated_features['Bezit woning'] = translated_features['Bezit woning'].map(person_home_ownership01).fillna(translated_features['Bezit woning'])
    translated_features['Intentie lening'] = translated_features['Intentie lening'].map(loan_intent01).fillna(translated_features['Intentie lening'])
    translated_features['Leningsgraad'] = translated_features['Leningsgraad'].map(loan_grade01).fillna(translated_features['Leningsgraad'])
    translated_features['Historische wantbetaling'] = translated_features['Historische wantbetaling'].map(cb_person_default_on_file01).fillna(translated_features['Historische wantbetaling'])
    # Show the data in a table form with headers
    st.dataframe(translated_features.transpose().dropna(), width=600, height=420)

# Create a button to show the text explanation with Lime
if st.button('Methode 2'):
    # Get the selected client's model
    model = models[selected_client]
    X_renamed = X.rename(columns=translations)
    # Get the data for the selected client
    client_data = creditriskclient.iloc[int(selected_client.split()[1]) - 1][features]
    # Make a Lime explainer
    explainer = LimeTabularExplainer(X_train.values, feature_names=X_renamed.columns, class_names=['0', '1'], discretize_continuous=True)
    # Make the text explanation
    text_explanation = text_explanation_lime(model, client_data)
    # Show the explanation in text
    st.write(text_explanation)

# Create a button to show the local explanation for the selected client
if st.button('Methode 3'):
    # Get the selected client's model
    model = models[selected_client]
    # Get the data for the selected client
    client_data = creditriskclient.iloc[int(selected_client.split()[1]) - 1][features]
    # Generate the LIME explanation for the selected client
    exp_lime = explain_client_lime(client_data, model)
    # Extract feature importance values and names from the explanation
    features_1 = [translations.get(x[0], x[0]) for x in exp_lime]
    importance = [x[1] for x in exp_lime]
    # Sort the importances
    sorted_indixes_importances_lime = np.argsort(importance)[::-1]
    # Sort the features and importance labels
    features_sorted = [features_1[i] for i in sorted_indixes_importances_lime]
    importance_sorted = [importance[i] for i in sorted_indixes_importances_lime]
    # Create a list of colors for the bars based on their values
    colors_sorted = ['red' if x < 0 else 'green' for x in importance_sorted]
    # Create a bar chart of the feature importance values with colors
    st.write('De rode balken zijn negatieve factoren en de groene balken zijn positieve factoren die invloed hadden op de leenstatus van deze specieke klant: ')
    fig, ax = plt.subplots()
    ax.barh(features_sorted, importance_sorted, color=colors_sorted)
    ax.set_xlabel('Invloed van factor')
    ax.set_ylabel('Factoren voor voorspelling')
    ax.set_title('Uitleg van factoren voor {}'.format(selected_client))
    # Show the graphic in Streamlit
    st.pyplot(fig)

# Create a button to show the model visualizations
if st.button('Methode 4'):
    # Get the selected client's model
    selected_model = models[selected_client]
    # Get the path for the selected client
    x = creditriskclient.loc[filtered_clients.index.values[0]]
    if isinstance(selected_model, DecisionTreeClassifier): 
        st.write('U gaat nu het gehele model met het pad van de specifieke klant zien!')
        show_decision_tree()
    elif isinstance(selected_model, RandomForestClassifier):
        st.write('Let op: dit is alleen het belangrijkste gedeelte (1/4) van het model en alleen het pad van deze specifieke klant!')
        show_random_forest()
    else: 
        st.write('Sorry, dit is geen boomvormig model.')

        # Working with counterfactuals

# Create an expandable container 
with st.expander("Methode 5"):
    st.write('Hieronder kunt u met behulp van de schuifbalken en het keuzemenu de waarde(n) van de factoren inkomen, percentage inkomen en leengraad wijzigen. Wanneer deze waarde(n) worden gewijzigd, wordt een nieuwe prognose gemaakt met de gewijzigde waarden van de klant')
    # Get the selected client's data from the creditriskclient dataframe
    selected_client_data = creditriskclient.loc[filtered_clients.index.values[0]]
    # Get the selected client's model
    model = models[selected_client]
    # Get the values for the sliders and the menu
    person_income_value = selected_client_data['person_income']
    loan_percent_income_value = selected_client_data['loan_percent_income']
    loan_grade_value = selected_client_data['loan_grade']
    # Create sliders to change the values of person_income, loan_percent_income, and loan_grade
    person_income_value = st.slider('Wat als de klant een ander inkomen zou hebben?', 
                                    min_value=float(0), 
                                    max_value=float(150000), # Can be changed to 2039784 because of client 10 
                                    value=float(person_income_value), 
                                    step=float(500))

    loan_percent_income_value = st.slider('Wat als het leenpercentage inkomen anders zou zijn?',
                                        min_value=float(0),
                                        max_value=float(1),
                                        value=float(loan_percent_income_value),
                                        step=float(0.05))
    # Create the selectbox, with the library from int to string 
    loan_grade_value = st.selectbox('Wat als de klant een andere leengraad zou hebben?', 
                                    sorted(loan_grade01.values()), 
                                    index=sorted(loan_grade01.values()).index(loan_grade01[loan_grade_value]))
    # Update the loan_grade_value variable to the numerical value
    loan_grade_value = list(loan_grade01.keys())[list(loan_grade01.values()).index(loan_grade_value)]
    # Create a copy of the selected client's data
    modified_client_data = selected_client_data.copy()
    # Update the values of person_income, loan_percent_income, and loan_grade based on the slider and selectbox values
    modified_client_data['person_income'] = person_income_value
    modified_client_data['loan_percent_income'] = loan_percent_income_value
    modified_client_data['loan_grade'] = loan_grade_value
    # Convert the modified client data to a format that can be used by the trained model
    modified_client_data = pd.DataFrame(modified_client_data).transpose()
    # Now you can index modified_client_data with the features list
    modified_client_data = modified_client_data[features]
    # Make a new prediction based on the modified features
    new_prediction = model.predict(modified_client_data)
    # Determine the prediction text based on the prediction
    if new_prediction[0] == 1:
        st.success(f'Op basis van deze waarden zou de klant wel zijn goedgekeurd!')
    else:
        prediction_str = 'nog steeds zijn afgekeurd'
        st.error(f'Op basis van deze waarden zou de klant zijn afgekeurd.')

            # Working with XAI-methods part 2

# Add a header + button  for working with counterfactuals
st.markdown("<h2 style='color: #0077c2;'>Deel 2: Gebruik van Explainable AI methoden</h2>", unsafe_allow_html=True)
st.write("Dit zijn XAI-methoden die model specifiek zijn.")

# Create a button to display the correlation figure
if st.button("Methode 6"):
    st.write("Correlatie verwijst naar een verband tussen twee factoren  die kunnen worden gemeten of waargenomen. Een positieve correlatie tussen twee factoren betekent dat wanneer de ene factor stijgt, de andere ook stijgt. Omgekeerd betekent een negatieve correlatie dat wanneer de ene factor stijgt, de andere factor daalt.")
    display_correlation_figure(selected_client)

# Create a button to show some information about the model
if st.button('Methode 7'):
    # Get the selected client's model
    model = models[selected_client]
    # Calculate the recall score
    recall = recall(model, X_test, y_test)
    recall_category = score_to_category(recall)
    st.write(f"De recall score is het aandeel van de feitelijke positieve gevallen die het model correct als positief heeft geïdentificeerd. Een hoge recall score betekent dat het model goed is in het oppikken van alle goedgekeurde leenaanvragen. In dit geval van dit specifieke model is de recall: {recall:.2f}. Dit betekent dat het model {recall_category} presteert.")
    # Calculate the certainly
    certainty = certainty(model, X_test)
    certainty_category = score_to_category(certainty)
    st.write(f"De zekerheidsscore geeft het vertrouwen van het model in zijn voorspellingen weer. Een hoge zekerheidsscore betekent dat het model veel vertrouwen heeft in zijn voorspellingen, terwijl een lage zekerheidsscore betekent dat het model minder vertrouwen heeft in zijn voorspellingen. In het geval van dit specifieke model is de zekerheidsscore: {certainty:.2f}. Dit betekent dat het model {certainty_category} presteert")

# Create a button to show the feature importance of the model
if st.button('Methode 8'):
    # Get the selected model for the selected client
    selected_model = models[selected_client]
    # Plot the feature importances for the selected client
    if isinstance(selected_model, DecisionTreeClassifier) or isinstance(selected_model, RandomForestClassifier):
        importances = selected_model.feature_importances_
        show_xlabel_ticks = True  # Show the X-axis tick labels for DecisionTree and RandomForest
    elif isinstance(selected_model, LogisticRegression):
        importances = np.abs(selected_model.coef_[0]) # Take absolute values of coefficients
        show_xlabel_ticks = False  # Hide the X-axis tick labels for LogisticRegression
    # Map feature names to translations
    feature_labels = [translations.get(feature, feature) for feature in features]    
    # Sort the importances
    sorted_indixes_importances = np.argsort(importances)[::-1]
    # Sort the importances and feature labels 
    sorted_importances = importances[sorted_indixes_importances]
    sorted_feature_labels = [feature_labels[i] for i in sorted_indixes_importances]

    # Make the graphic with the sorted arrays
    st.write("Feature importance verwijst naar een methode die wordt gebruikt bij gegevensanalyse of AI om te bepalen welke factoren (features) of variabelen in een dataset het nuttigst of invloedrijkst zijn bij het voorspellen van een bepaalde uitkomst of doelvariabele. Dus in dit geval zijn in het figuur de meest invloedrijke factoren te zien voor het voorspellen van de leenstatus. Let op: de feature importance is per model verschillend.")
    plt.barh(range(len(sorted_feature_labels)), sorted_importances)
    plt.yticks(range(len(sorted_feature_labels)), sorted_feature_labels)
    plt.gca().invert_yaxis() # invert the y-axis to show the features in descending order
    plt.ylabel('Factoren')
    plt.title('Belangrijkste factoren voor de voorspelling')
    if show_xlabel_ticks:
        plt.xlabel('Belangrijkheidsgraad')
    else:
        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # hide the X-axis tick labels
        plt.xlabel('Belangrijkheidsgraad', labelpad=10) # add separate x-axis label
    # Show the graphic in Streamlit
    st.pyplot(plt)
