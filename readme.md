# P2Pump

## Overview
P2Pump is a decentralized wellness and fitness application that leverages **Peer-to-Peer (P2P) networking** and **Federated Learning** to provide personalized fitness recommendations while ensuring user privacy. The app collects user activity data (step count, mood, and calories burnt) and predicts daily activity goals without transferring raw data to a central server.

## Features
- **Decentralized (P2P) Architecture**: Uses **Py2P** to enable direct communication between edge devices.
- **Federated Learning**: Trains models locally using **PyTorch** and securely shares only model updates.
- **Multi-Cloud Infrastructure**:
  - **AWS EC2**: Emulates edge devices and hosts local models.
  - **Azure ML**: Aggregates model updates and serves the global model.
  - **Google Cloud Storage**: Backs up model weights and metadata.
- **Firebase Authentication**: Secure user authentication and data management.

## Technologies Used
- **Programming Languages**: Python, Dart (Flutter for front-end) ➡️ [Link to frontend code](https://github.com/apoorvsingh2000/P2Pump)
- **Frameworks & Libraries**:
  - PyTorch (ML Model Training)
  - Py2P (P2P Communication)
  - Firebase (Authentication)
- **Cloud Services**:
  - AWS EC2 (Edge Device Emulation)
  - Azure ML (Global Model Aggregation)
  - Google Cloud Storage (Model Backup)

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Py2P
- Firebase SDK
- Google Cloud SDK

## Deployment
- **Edge Devices**: Deployed on AWS EC2 instances.
- **Global Model**: Hosted on Azure ML.
- **Backup & Storage**: Managed using Google Cloud Storage.

## Future Enhancements
- Extend support for additional wearables (Apple Watch, Fitbit, Samsung Wear).
- Implement a real-time analytics dashboard.
- Improve model personalization through reinforcement learning.

## Contributors
- **Raghav Mantri**
- **Apoorv Singh**
- **Divyanshu Sharma**

