import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Read the Excel file
df = pd.read_excel('../resources/Cardiology.xlsx',sheet_name='Sheet1')

# Recording ratios of each distinct option in the 'class' column
survived_ratios = df['class'].value_counts(normalize=True).to_dict()
print("Original dataframe ratios")
print(survived_ratios)

# Split the DataFrame into features (X) and target (y)
X = df.drop('class', axis=1) # Features
y = df['class'] # Target variable

# Perform label encoding on the categorical features
categorical_cols = ['age', 'sex', 'chest pain type', 'blood pressure', 'cholesterol', 
                    'Fasting blood sugar <120', 'resting ecg', 'maximum heart rate', 'angina', 
                    'peak', 'slope', '#colored vessels', 'thal']
label_encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Split the data into train and test sets within the model
#clf = DecisionTreeClassifier(criterion='gini')
clf = DecisionTreeClassifier(criterion='entropy') #higher accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

train_survived_ratios = y_train.value_counts(normalize=True).to_dict()
print("Survived Attribute Ratios in the Training Set:")
print(train_survived_ratios)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)