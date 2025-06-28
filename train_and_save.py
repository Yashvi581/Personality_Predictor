# train_and_save.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load dataset
dataset = pd.read_csv("D:/streamlit/personality_app/personality_datasert.csv")

# Encode categorical features
le = LabelEncoder()
dataset['Stage_fear'] = le.fit_transform(dataset['Stage_fear'])
dataset['Drained_after_socializing'] = le.fit_transform(dataset['Drained_after_socializing'])
dataset['Personality'] = le.fit_transform(dataset['Personality'])

# Split
x = dataset.drop(['Personality'], axis=1)
y = dataset['Personality']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Build model
model = Sequential([
    Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=0)

# Save model and scaler
model.save("personality_model.h5")
joblib.dump(scaler, "scaler.joblib")

print("✅ Model and scaler saved!")
