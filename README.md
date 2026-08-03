# 🍽️ Food Recommendation System

A Machine Learning-based Food Recommendation System built using **Python**, **Streamlit**, and the **Apriori Algorithm**. The application recommends food items based on customer purchase patterns and provides a simple, interactive food ordering experience.

---

## 📌 Overview

This project uses **Association Rule Mining (Apriori Algorithm)** to identify frequently ordered food combinations and recommend relevant food items to customers. Users can browse food items, filter by category and region, add items to a shopping cart, calculate delivery charges, place orders, and retrain the recommendation model using new customer transactions.

---

## ✨ Features

- 🔍 Search food items
- 🍽️ Category & Region-based filtering
- 🤖 Apriori-based food recommendations
- 🛒 Shopping cart with quantity management
- 🚚 Dynamic delivery charge calculation
- 💰 Order summary with grand total
- 📜 Customer order history
- 📈 Retrain recommendation model using customer orders
- ⭐ Interactive Streamlit interface

---

## 🧠 Machine Learning

- **Algorithm:** Apriori
- **Technique:** Association Rule Mining
- **Library:** MLxtend
- **Purpose:** Recommend food items that are frequently purchased together.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- MLxtend
- OpenPyXL

---

## 📂 Project Structure

```text
Food-Recommendation-System/
│
├── app.py
├── customer_orders.csv
├── Apriori_500_Realistic_Transactions.csv
├── requirements.txt
├── README.md
└── assets/
```

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/Food-Recommendation-System.git

cd Food-Recommendation-System

pip install -r requirements.txt

streamlit run app.py
```

---

## 📊 Project Workflow

1. Load Food Dataset
2. Apply Category & Region Filters
3. Generate Frequent Itemsets using Apriori
4. Recommend Related Food Items
5. Add Items to Cart
6. Calculate Delivery Charges
7. Place Customer Orders
8. Save Order History
9. Retrain the Recommendation Model

---

## 🎯 Future Enhancements

- User Authentication
- Online Payment Integration
- Live Delivery Tracking
- Personalized Recommendations
- Admin Dashboard
- Food Images
- Customer Reviews & Ratings

---

## 👨‍💻 Author

**Yogeshwar Tribhuvan**  
B.Tech (Artificial Intelligence & Machine Learning)

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
