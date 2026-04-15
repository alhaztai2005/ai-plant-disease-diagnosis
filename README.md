# ai-plant-disease-diagnosis
**Plant Disease Detection with RAG Chatbot**

This project combines machine learning and Retrieval-Augmented Generation (RAG) to detect diseases in tomato plants and provide intelligent, context-aware assistance. It uses a TensorFlow/Keras-based Convolutional Neural Network (CNN) for image-based disease classification, along with a RAG chatbot that retrieves relevant information from a knowledge dataset to answer user queries.

The system helps farmers and gardeners not only identify plant diseases from images but also get detailed insights about symptoms, causes, and treatments, enabling timely intervention and reducing crop losses.

-------------------------------------------------------------------------------------------

*Key Feature*

  >Disease Detection: Accurately classifies a range of tomato plant diseases from images.

  >CNN Architecture: Employs deep learning using Convolutional Neural Networks for high 
   precision.

  >User-Friendly: Simple scripts for training, evaluation, and prediction.

  >Scalable Design: Easily extendable to support disease detection in other crops or plants.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*How It Works*
>Input: User uploads an image of a tomato plant.

>Preprocessing: Image is resized and normalized to fit the CNN’s input format.

>Prediction: The trained model analyzes the image and predicts the disease class or 
 identifies the plant as healthy.

>Output: The prediction is displayed along with confidence scores.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Model Architecture*
>The CNN model includes the following layers:

>Input Layer: Accepts images of shape (128, 128, 3).

>Convolutional Layers: Feature extraction using Conv2D layers with Batch Normalization and MaxPooling.

>Global Average Pooling: Reduces spatial dimensions while preserving key features.

>Dense Layers: Fully connected layers with Dropout to prevent overfitting.

>Output Layer: Softmax activation for multi-class classification.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Dataset*

The model is trained on the Tomato Plant Disease Dataset, which includes the following classes:

    Bacterial Spot
    
    Early Blight
    
    Late Blight
    
    Leaf Mold
    
    Septoria Leaf Spot
    
    Spider Mites (Two-Spotted Spider Mite)
    
    Target Spot
    
    Tomato Mosaic Virus
    
    Tomato Yellow Leaf Curl Virus
    
    Healthy Plants

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Performance Metrics*

>Training Accuracy: 96.15%

>Training Loss: 0.117

>Validation Accuracy: 96.20%

>Validation Loss: 0.166

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

RAG Chatbot Integration (New Feature)

This project also includes a Retrieval-Augmented Generation (RAG) based chatbot to provide intelligent and context-aware responses related to plant diseases.

Key Features

Combines retrieval + generation for accurate answers
Uses document embeddings to fetch relevant plant disease information
Reduces hallucination by grounding responses in dataset knowledge
Can answer queries like symptoms, treatments, and prevention methods

Dataset (RAG)

FarmGenie QnA Dataset.csv – Custom dataset used for retrieval of plant disease related questions and answers.

How It Works

User Query → User asks a question
Retriever → Searches relevant information from dataset using embeddings
Generator (LLM) → Generates response based on retrieved context
Final Output → Accurate and contextual answer

Tech Stack

Embedding Model: all-MiniLM-L6-v2
Vector Store: FAISS / ChromaDB
Backend: Python (Flask)
LLM: OpenAI / local model

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*How to Use*
1. Clone the Repository
   git clone https://github.com/Rutu2601/Plant_Disease_Detection.git
   cd Plant_Disease_Detection
2. Install Dependencies
   pip install -r requirements.txt
3. Prepare the Dataset
   Structure your dataset as follows:


![image](https://github.com/user-attachments/assets/11ff6b6b-990d-4141-85fe-6779b48e3f5e)

4. Train the Model
    Open train.ipynb in Jupyter Notebook and run all cells to:
    
    Preprocess the data
    
    Train the model
    
    Save the trained model as bestModel.keras

5. Run the Flask App
    Start the web app using:
    python app.py
    Then open your browser and go to: http://127.0.0.1:5000
    Upload a tomato plant image through the interface to see the predicted disease.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Project Structure*

![image](https://github.com/user-attachments/assets/6aa63aad-ba56-478c-a3a6-1523c0134f33)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Acknowledgments*

Dataset sourced from PlantVillage
