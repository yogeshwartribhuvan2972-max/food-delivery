import streamlit as st
from streamlit_searchbox import st_searchbox
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os
from datetime import datetime
LAST_TRAINED_FILE = "last_trained_order.txt"

if not os.path.exists(LAST_TRAINED_FILE):
    with open(LAST_TRAINED_FILE, "w") as f:
        f.write("0")

# ---------------------------------
# Page Settings
# ---------------------------------
st.set_page_config(
    page_title="Food Recommendation System",
    page_icon="🍽️",
    layout="centered"
)

# ---------------------------------
# Custom CSS (clean, professional look)
# ---------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        max-width: 480px;
    }

    /* Card style */
    .food-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
    }
    .food-emoji {
        font-size: 34px;
        background: #f6f6f6;
        border-radius: 12px;
        width: 56px;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 14px;
    }
    .food-name {
        font-weight: 600;
        font-size: 16px;
        margin: 0;
        color: #222;
    }
    .food-price {
        color: #444;
        font-size: 14px;
        margin: 0;
    }
    .food-rating {
        color: #888;
        font-size: 12px;
        margin: 0;
    }
    .section-title {
        font-weight: 700;
        font-size: 15px;
        color: #e53935;
        letter-spacing: 0.5px;
        margin: 18px 0 10px 2px;
    }

    /* Buttons -> red rounded squares like the reference image */
    div.stButton > button {
    background-color: #e53935;
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    width: 52px;
    height: 52px;
    font-size: 22px;
    padding: 0;
}
    div.stButton > button:hover {
        background-color: #c62828;
        color: white;
    }

    /* "+" (add to cart / increase qty) buttons -> green */
    div[class*="st-key-plus_"] button,
    div[class*="st-key-plus2_"] button {
        background-color: #43a047;
    }
    div[class*="st-key-plus_"] button:hover,
    div[class*="st-key-plus2_"] button:hover {
        background-color: #2e7d32;
    }
    /* Cart Button */
div[class*="st-key-cart_btn"] button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    background-color: #6C63FF;
    color: white;
}

div[class*="st-key-cart_btn"] button:hover {
    background-color: #5A52E0;
    color: white;
}
/* Use Orders for Training Button */
div[class*="st-key-train_btn"] button {
    width: 100%;
    height: 60px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    background-color: #ff9800;
    color: white;
}

div[class*="st-key-train_btn"] button:hover {
    background-color: #f57c00;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🍽️ Food Recommendation System")
st.caption("Personalized recommendations powered by the Apriori algorithm")

st.sidebar.header("⚙️ Recommendation Settings")

min_support = st.sidebar.number_input(
    "Minimum Support",
    min_value=0.001,
    max_value=1.0,
    value=0.02,
    step=0.01,
    format="%.3f"
)
min_confidence = st.sidebar.number_input(
    "Minimum Confidence",
    min_value=0.001,
    max_value=1.0,
    value=0.02,
    step=0.01,
    format="%.3f"
)

top_n = st.sidebar.number_input(
    "Number of Recommendations", min_value= 1, max_value=5, value=2, step=1
)

# cart is now {item_name: quantity} so we can show a quantity stepper
if "cart" not in st.session_state:
    st.session_state.cart = {}
if "history" not in st.session_state:
    st.session_state.history = []    
if "show_details" not in st.session_state:
    st.session_state.show_details = False
if "show_cart" not in st.session_state:
        st.session_state.show_cart = False
if "last_selected" not in st.session_state:
    st.session_state.last_selected = None
if "reset_widgets" not in st.session_state:
    st.session_state.reset_widgets = False
# ---------------------------------
# Load Master Dataset
# ---------------------------------

master_df = pd.read_excel("Food_Items_Master_Dataset_Updated.xlsx")

# Item name index banva
master_df.set_index("Item", inplace=True)
# ---------------------------------
# Load Apriori Transactions
# ---------------------------------

transaction_df = pd.read_csv("Apriori_500_Realistic_Transactions.csv")

transactions = (
    transaction_df["Items"]
    .apply(lambda x: [item.strip() for item in x.split(",")])
    .tolist()
)
# print(transaction_df.head())
# print(transaction_df.shape)
# print(transactions[:5])
# print("Duplicate Items:", master_df.index[master_df.index.duplicated()].tolist())
# ---------------------------------
# Encoding + Apriori (unchanged logic)
# ---------------------------------
te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)
df = pd.DataFrame(encoded, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
# print("Frequent Itemsets:", frequent_itemsets.shape)
# print("Rules:", rules.shape)
# print(rules.columns)
# print(rules.head())
# print("Rules Shape:", rules.shape)
# print("Rules Columns:", rules.columns.tolist())

# def recommend(item):
#     rec = rules[rules["antecedents"].apply(lambda x: item in x)]
#     rec = rec.sort_values(by="confidence", ascending=False)
#     return rec
def recommend(item):

    # Rules nahi banle tar
    if rules.empty or "confidence" not in rules.columns:
        return pd.DataFrame()

    rec = rules[rules["antecedents"].apply(lambda x: item in x)]

    # Ya item sathi recommendation nahi
    if rec.empty:
        return pd.DataFrame()

    rec = rec.sort_values(by="confidence", ascending=False)

    return rec

def add_to_cart(item):
    st.session_state.cart[item] = st.session_state.cart.get(item, 0) + 1


def remove_from_cart(item):
    if item in st.session_state.cart:
        st.session_state.cart[item] -= 1
        if st.session_state.cart[item] <= 0:
            del st.session_state.cart[item]
    # Cart completely empty zala tar recommendation remove kara
    if len(st.session_state.cart) == 0:
        st.session_state.last_selected = None

def render_item_row(item, key_prefix):
    """Renders a food card with an Add (+) button, or a -/qty/+ stepper
    once the item is already in the cart. Used for the selected item
    and recommended items so behaviour stays consistent."""
    qty = st.session_state.cart.get(item, 0)

    if qty == 0:
        col1, col2 = st.columns([6, 1], vertical_alignment="center")
        with col1:
            st.markdown(food_card_html(item), unsafe_allow_html=True)
        with col2:
            if st.button("➕", key=f"plus_{key_prefix}_{item}"):
                add_to_cart(item)

                # Selected item remember kara
                st.session_state.last_selected = item

                # Search ani filters reset
                st.session_state.last_selected = item
                st.session_state.reset_widgets = True
                st.rerun()
    else:
        col1, col2, col3, col4 = st.columns([5, 1, 1, 1], vertical_alignment="center")
        with col1:
            st.markdown(food_card_html(item), unsafe_allow_html=True)
        with col2:
            if st.button("➖", key=f"minus_{key_prefix}_{item}"):
                remove_from_cart(item)
                st.rerun()
        with col3:
            st.markdown(f"<p style='text-align:center; font-weight:600;'>{qty}</p>", unsafe_allow_html=True)
        with col4:
            if st.button("➕", key=f"plus2_{key_prefix}_{item}"):
                add_to_cart(item)
                st.rerun()


def food_card_html(item):

    food = master_df.loc[item]
    # print("Selected Item :", repr(item))
    # print("Food Type :", type(food))
    # print(food)

    return (
        '<div class="food-card">'
        '<div style="display:flex; align-items:center;">'

        f'<div class="food-emoji">{food["Emoji"]}</div>'

        '<div>'

        f'<p class="food-name">{item}</p>'

        f'<p class="food-price">₹{food["Price"]}</p>'

        f'<p class="food-rating">📦 {food["Quantity"]}</p>'

        f'<p class="food-rating">⭐ {food["Rating"]}</p>'

        '</div>'

        '</div>'

        '</div>'
    )
# ---------------------------------
# Dynamic Category & Region Filters
# ---------------------------------

category_list = ["All"] + sorted(master_df["Category"].unique())

col1, col2 = st.columns(2)

# First Category
with col1:
    selected_category = st.selectbox(
        "🍽️ Category",
        category_list
    )

# Category nusar Region update hoil
temp_df = master_df.copy()

if selected_category != "All":
    temp_df = temp_df[
        temp_df["Category"] == selected_category
    ]

region_list = ["All"] + sorted(temp_df["Region"].unique())

# Then Region
with col2:
    selected_region = st.selectbox(
        "🌍 Region",
        region_list
    )
# ---------------------------------
# Apply Filters
# ---------------------------------

filtered_df = master_df.copy()

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == selected_region
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == selected_category
    ]

# ============================================================
# SELECT FOOD (Filtered Menu)
# ============================================================

filtered_items = sorted(filtered_df.index.unique().tolist())


def search_food(search_term: str):

    if not search_term:
        return filtered_items

    return sorted(
        [
            item
            for item in master_df.index.unique()
            if search_term.lower() in item.lower()
        ]
    )


selected = st_searchbox(
    search_function=search_food,
    label="🍔 Choose Food",
    placeholder="Search or select food...",
    key="food_box"
)


# ============================================================
# SELECTED ITEM
# ============================================================

# Search madhun item select zala tar tyalach priority
# if search_selected is not None:
#     selected = search_selected

display_item = selected

# Item cart madhye gela ki Selected Item hide kara
if (
    display_item is not None
    and display_item in st.session_state.cart
):
    display_item = None

if display_item is not None:

    st.markdown(
        '<p class="section-title">SELECTED ITEM</p>',
        unsafe_allow_html=True
    )

    # Search madhun aala pan current filters madhye nasel tar info dakhva
    if display_item not in filtered_df.index:

        food = master_df.loc[display_item]

        st.info(
            f"""
📍 Region : {food['Region']}

🍽 Category : {food['Category']}

This item is outside your selected filters.
"""
        )

    render_item_row(display_item, key_prefix="sel")

# -----------------------------
# Recommendations ("You May Also Like")
# -----------------------------
if st.session_state.last_selected is not None:

    result = recommend(st.session_state.last_selected)

    if result.empty:

        st.info("🍽️ Discover something new from our menu!")

    else:

        st.markdown(
            '<p class="section-title">🧡 YOU MAY ALSO LIKE</p>',
            unsafe_allow_html=True
        )

        recommendations = []

        for _, row in result.iterrows():

            for rec_item in list(row["consequents"]):

                rec_item = str(rec_item).strip()

                if rec_item == st.session_state.last_selected:
                    continue

                if rec_item not in master_df.index:
                    continue

                if rec_item not in recommendations:
                    recommendations.append(rec_item)

        recommendations = recommendations[:top_n]

        for rec_item in recommendations:

            # Cart madhye asel tar recommendation hide kara
            if rec_item in st.session_state.cart:
                continue

            render_item_row(rec_item, key_prefix=f"rec_{rec_item}")

else:
    result = pd.DataFrame()
# -----------------------------
# Cart Button
# -----------------------------
cart_items = sum(st.session_state.cart.values())
cart_total = sum(
    master_df.loc[item, "Price"] * qty
    for item, qty in st.session_state.cart.items()
)

if st.button(
    f"🛒 Cart ({cart_items} items) • ₹{cart_total}",
    use_container_width=True,
    key="cart_btn"
):
    st.session_state.show_cart = not st.session_state.show_cart
    st.rerun()
# -----------------------------
# Show Cart Only When Clicked
# -----------------------------
if st.session_state.show_cart:
    if "last_selected" not in st.session_state:
        st.session_state.last_selected = None

    st.markdown("### 🛒 Your Cart")

    if len(st.session_state.cart) == 0:
        st.info("Your cart is empty.")

    else:

        total = 0
    
        for product, qty in list(st.session_state.cart.items()):
            render_item_row(product, key_prefix="cart")
            total += master_df.loc[product, "Price"] * qty

        # -------------------------
        # Order Summary
        # -------------------------

        total_price = total
        total_items = sum(st.session_state.cart.values())

        st.subheader("🧾 Order Summary")

        st.write(f"🛒 Total Items : {total_items}")
        st.write(f"💰 Food Total : ₹{total_price}")

        distance = st.radio(
            "🚚 Delivery Distance",
            ["500 m", "1 km", "1.5 km", "2 km", "Above 2 km"],
            horizontal=True
        )

        # -------------------------
        # Delivery Charge
        # -------------------------

        if total_price <= 100:

            delivery = {
                "500 m": 20,
                "1 km": 30,
                "1.5 km": 40,
                "2 km": 50,
                "Above 2 km": 100
            }[distance]

        elif total_price <= 200:

            delivery = {
                "500 m": 15,
                "1 km": 25,
                "1.5 km": 35,
                "2 km": 45,
                "Above 2 km": 100
            }[distance]

        elif total_price <= 300:

            delivery = {
                "500 m": 10,
                "1 km": 20,
                "1.5 km": 30,
                "2 km": 40,
                "Above 2 km": 100
            }[distance]

        elif total_price <= 500:

            delivery = {
                "500 m": 5,
                "1 km": 15,
                "1.5 km": 25,
                "2 km": 35,
                "Above 2 km": 100
            }[distance]

        else:

            if distance == "Above 2 km":
                delivery = 100
            else:
                delivery = 0

        # -------------------------
        # Discount
        # -------------------------

        discount = 0

        if total_price >= 1200:
            discount = 100

        elif total_price >= 800:
            discount = 50

        # -------------------------
        # Grand Total
        # -------------------------

        grand_total = total_price + delivery - discount

        st.caption("📍 Delivery distance is estimated based on your selected map location.")

        if total_price < 100:
            st.info(
                "💡 Add a few more items to your cart for better savings on delivery."
            )

        elif total_price < 300:

            st.info(
                "🍽️ Add one more item to reduce your delivery charges."
            )

        elif total_price < 400:

            st.info(
                "🎁 Place a bigger order next time to enjoy FREE Delivery."
            )

        elif total_price < 500:

            st.warning(
                f"🚚 Add just ₹{500-total_price} more to unlock FREE Delivery!"
            )

        else:

            if distance != "Above 2 km":
                st.success(
                    "🎉 Congratulations! You unlocked FREE Delivery."
                )

        st.write(f"🚚 Delivery Charge : ₹{delivery}")
        st.write(f"🎁 Discount : ₹{discount}")

        st.markdown("---")

        st.success(f"💵 Grand Total : ₹{grand_total}")

        # -------------------------
        # Buy Now
        # -------------------------

        if st.button("🛍️ Buy Now", use_container_width=True):
            

            if st.session_state.cart:
                st.session_state.last_selected = None
                st.session_state.reset_widgets = True
                st.session_state.show_cart = False

                order_file = "customer_orders.csv"

                # Order Items
                order_items = ",".join(st.session_state.cart.keys())

                # Total Quantity
                total_items = sum(st.session_state.cart.values())

                # Date Time
                now = datetime.now()

                order_date = now.strftime("%Y-%m-%d")
                order_time = now.strftime("%I:%M:%S %p")

                # Order ID
                if os.path.exists(order_file):

                    existing = pd.read_csv(order_file)

                    order_id = len(existing) + 1

                else:

                    order_id = 1

                # New Order
                new_order = pd.DataFrame({

                    "Order_ID":[order_id],

                    "Order_Date":[order_date],

                    "Order_Time":[order_time],

                    "Order_Items":[order_items],

                    "Total_Items":[total_items],

                    "Food_Total":[total_price],

                    "Distance":[distance],

                    "Delivery_Charge":[delivery],

                    "Discount":[discount],

                    "Grand_Total":[grand_total]

                })

                # Append

                new_order.to_csv(
                    order_file,
                    mode="a",
                    header=not os.path.exists(order_file),
                    index=False
                )

                # Save in Session History (temporary)
                order = {
                    "items": st.session_state.cart.copy(),
                    "food_total": total_price,
                    "delivery": delivery,
                    "discount": discount,
                    "grand_total": grand_total
                }

                st.session_state.history.append(order)

                # Empty Cart
                st.session_state.cart = {}

                st.success(f"✅ Order Placed Successfully!\n\n💵 Amount Paid : ₹{grand_total}")

                st.rerun()
            else:
                st.warning("Your cart is empty!")
st.markdown("## 📜 Order History")

if st.button(
"🔄 Use Orders for Training",
use_container_width=True,
key="train_btn"
):
    with open(LAST_TRAINED_FILE, "r") as f:
        last_trained = int(f.read())
    # Load both files
    orders = pd.read_csv("customer_orders.csv")
    training = pd.read_csv("Apriori_500_Realistic_Transactions.csv")

    # Keep only Order_Items column
    new_orders = orders[orders["Order_ID"] > last_trained]

    if new_orders.empty:
        st.info("No new orders available for training.")
        st.stop()

    new_transactions = new_orders[["Order_Items"]].copy()
    new_transactions.columns = ["Items"]

    # Rename to match training dataset
    new_transactions.columns = ["Items"]

    # Append
    updated_training = pd.concat(
        [training, new_transactions],
        ignore_index=True
    )
    # Save back
    updated_training.to_csv(
        "Apriori_500_Realistic_Transactions.csv",
        index=False
    )
    latest_order = orders["Order_ID"].max()

    with open(LAST_TRAINED_FILE, "w") as f:
        f.write(str(latest_order))

    st.success("✅ Customer orders added to the Apriori training dataset.")
if os.path.exists("customer_orders.csv"):

    orders = pd.read_csv("customer_orders.csv")

    if orders.empty:
        st.info("No orders yet.")

    else:
        st.dataframe(
            orders,
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("No orders yet.")
    # -----------------------------
# Show Details button
# -----------------------------
st.divider()
if st.button("📊Show Detail" if not st.session_state.show_details else "🔽 Hide Details"):
    st.session_state.show_details = not st.session_state.show_details
    st.rerun()

if st.session_state.show_details:
    if not result.empty:
        top_rule = result.iloc[0]
        st.write(f"**Support :** {top_rule['support']:.2f}")
        st.write(f"**Confidence :** {top_rule['confidence']:.2f}")
        st.write(f"**Lift :** {top_rule['lift']:.2f}")

    st.markdown("### 📈 Top Frequent Itemsets")
    show = frequent_itemsets.copy()
    show["Itemsets"] = show["itemsets"].astype(str)
    st.dataframe(
        show[["Itemsets", "support"]],
        use_container_width=True,
        hide_index=True
    )
