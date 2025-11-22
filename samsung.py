import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from io import StringIO
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Samsung Pricing Strategy Engine",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR DASHBOARD LOOK ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .surplus-card {
        background-color: #ffffff;
        border-left: 5px solid #22c55e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .price-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        transition: transform 0.2s;
    }
    .price-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .bundle-card {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    .insight-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DEFAULT DATA LOADING ---
@st.cache_data
def load_default_data():
    csv_content = """Samsung_Smartphone,Samsung_Smart_TV_43in,Samsung_Smart_Watch,Samsung_Washing_Machine,Samsung_AC_1.5_Tonne
46371,38390,10695,22633,46883
95402,37411,5179,28348,54920
32191,55430,19183,29138,60153
114921,47641,28643,48809,56291
46083,50986,13906,50804,49542
64405,55260,12398,29402,26565
103194,49171,17219,43524,38325
23787,58884,30076,46407,41890
93946,41870,17478,33432,46575
58329,62791,23984,47980,45185
68996,32708,17702,56427,49056
32188,52276,15916,53980,46529
55920,64810,17258,42960,54298
66398,39468,25363,33289,32665
50826,42141,24927,29293,67821
32615,41404,17073,57167,58115
38382,49750,13900,30765,44412
84204,44885,10859,34863,46128
20665,32725,3779,30417,64284
69383,64890,10091,36191,62398
117683,37521,11550,24967,46031
14969,48720,23064,45343,33189
33053,50829,27022,36404,66709
66450,27818,24401,34430,52347
24070,31982,23999,33078,53377
59917,72289,37707,38257,42576
67266,53457,34118,44263,57226
59797,27095,13925,49530,50043
45257,59211,11026,46647,48000
68386,47874,11250,35024,54738
60248,30620,30173,46334,45454
36176,54107,9741,36971,46979
57996,47575,15294,39824,52310
87763,37532,19157,40865,51989
65853,63974,21090,33966,64907
51822,40906,20662,35954,50164
42733,40659,25943,41403,44002
26835,40227,20095,31647,42982
38412,57901,28261,43126,49510
16075,55157,9129,32851,36597
109151,65738,28578,38405,50746
37095,38547,20525,46484,45953
79637,58819,12948,19569,44899
33590,50056,35462,29707,39715
40600,41517,32533,37780,48608
48451,27292,28054,24772,31506
42894,60501,30957,28602,51197
77875,39015,16093,43198,42923
100704,52191,13057,19820,60756
86872,49220,12011,39054,49334
32512,22583,29193,20544,55947
75670,76553,27336,25214,58427
32095,60253,35800,33743,59563
92391,47160,28152,53704,59209
54864,54090,21559,37396,40926
43581,46846,10948,31763,43472
32763,56840,11555,34240,53787
16242,34194,17935,37974,49593
38604,45902,25344,39308,51248
92493,42434,11771,30527,52969
77761,46314,31159,28325,36337
103696,46029,22897,36171,38760
28234,34650,14775,40011,51945
60093,72973,29720,45635,51637
39130,24820,20731,38596,57181
56579,23887,24289,41398,53398
42669,43667,16157,35905,40285
40592,51765,16315,37921,33878
58761,36415,5366,39748,45973
34595,48690,19325,29584,45437
51650,55871,25125,51387,55531
64566,31878,34629,32629,52805
30566,74280,30015,38911,59275
95286,42473,9214,37775,48817
54048,52596,35755,47155,43021
80148,32887,11146,25111,48817
64812,55300,7272,48587,40570
69482,61695,24760,26104,33252
70019,32566,30669,26405,48709
86847,33740,33184,47551,46570
61688,40532,27276,30211,31021
40302,24231,12538,39101,54161
28121,74587,11269,35435,52109
61135,50958,8116,51400,34111
24797,51293,20335,44342,53845
38218,70716,18221,32363,34652
24153,61218,9965,45512,49132
56993,60249,17750,29255,51561
33002,42566,19994,39203,48693
61447,42411,23991,43282,54922
77100,40919,16721,41725,45826
80408,42804,14889,40347,40478
42051,43542,22414,46169,48392
56998,34491,22909,37655,44621
57987,43106,13957,51021,38045
51892,45914,5966,26416,48021
64909,41541,32912,32892,40053
36330,35748,24954,44375,48001
106484,43666,12449,31650,50849
27990,46983,19956,34966,55999"""
    return pd.read_csv(StringIO(csv_content))

# --- 2. OPTIMIZATION LOGIC (Cached) ---

def calculate_baseline_separate(df, products):
    """Calculates optimal revenue if products are sold purely separately."""
    total_rev = 0
    prices = {}
    for prod in products:
        wtp = df[prod].values
        best_p = 0
        best_r = 0
        # Simple Grid Search on existing WTP points
        candidates = np.unique(wtp)
        for p in candidates:
            rev = p * np.sum(wtp >= p)
            if rev > best_r:
                best_r = rev
                best_p = p
        prices[prod] = best_p
        total_rev += best_r
    return total_rev, prices

@st.cache_data(show_spinner=False)
def run_evolutionary_optimization(df, products):
    """Runs Differential Evolution to find optimal Mixed Bundling prices."""
    
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_products = len(products)

    def revenue_objective(prices):
        # prices = [p1, p2, p3, p4, p5, p_bundle]
        indiv_prices = np.array(prices[:n_products])
        bundle_price = prices[n_products]

        # 1. Surplus if buying individually
        surplus_matrix = np.maximum(wtp_matrix - indiv_prices, 0)
        surplus_indiv = np.sum(surplus_matrix, axis=1)
        
        # 2. Revenue if buying individually (only for items bought)
        buy_flags = (wtp_matrix >= indiv_prices)
        revenue_indiv = np.sum(buy_flags * indiv_prices, axis=1)

        # 3. Surplus if buying bundle
        surplus_bundle = bundle_sum_values - bundle_price

        # 4. Decision
        # Buy Bundle if Surplus_Bundle >= Surplus_Indiv AND Surplus_Bundle >= 0
        buy_bundle_mask = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        buy_indiv_mask = (~buy_bundle_mask) & (surplus_indiv > 0)

        total_rev = np.sum(buy_bundle_mask * bundle_price) + np.sum(revenue_indiv[buy_indiv_mask])
        return -total_rev # Minimize negative revenue

    # Bounds: 0 to 1.5x Max WTP (to allow high anchors)
    bounds = []
    for i in range(n_products):
        bounds.append((0, np.max(wtp_matrix[:, i]) * 1.5))
    bounds.append((0, np.max(bundle_sum_values))) # Bundle Price

    # Optimization
    result = differential_evolution(
        revenue_objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=100, 
        popsize=15, 
        tol=0.01, 
        seed=42
    )

    return result.x, -result.fun

def generate_customer_details(df, products, opt_prices):
    """Generates detailed table of customer decisions based on optimized prices."""
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_products = len(products)
    
    indiv_prices = opt_prices[:n_products]
    bundle_price = opt_prices[n_products]

    results = []

    for i in range(len(df)):
        # Indiv Surplus
        row_wtp = wtp_matrix[i]
        surplus_items = np.maximum(row_wtp - indiv_prices, 0)
        surplus_indiv = np.sum(surplus_items)
        
        # Indiv Revenue/Items
        items_bought = []
        cost_indiv = 0
        for j, p in enumerate(products):
            if row_wtp[j] >= indiv_prices[j]:
                items_bought.append(p)
                cost_indiv += indiv_prices[j]
        
        # Bundle Surplus
        surplus_bundle = bundle_sum_values[i] - bundle_price
        
        decision = ""
        revenue = 0
        final_surplus = 0
        items_str = ""

        if surplus_bundle >= surplus_indiv and surplus_bundle >= 0:
            decision = "Bundle"
            revenue = bundle_price
            final_surplus = surplus_bundle
            items_str = "Full Ecosystem"
        elif surplus_indiv > 0:
            decision = "Individual"
            revenue = cost_indiv
            final_surplus = surplus_indiv
            items_str = ", ".join([p.replace("Samsung_", "") for p in items_bought])
        else:
            decision = "None"
            revenue = 0
            final_surplus = 0
            items_str = "-"
            
        results.append({
            "Customer ID": i+1,
            "Decision": decision,
            "Items": items_str,
            "Revenue": revenue,
            "Surplus": final_surplus
        })
        
    return pd.DataFrame(results)

# --- MAIN APP UI ---

def main():
    st.title("💎 Samsung Pricing Strategy Optimization")
    st.markdown("This dashboard uses **Evolutionary Algorithms** (Differential Evolution) to find the optimal Mixed Bundling strategy.")

    # Sidebar / File Upload
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Upload WTP Data (CSV)", type=['csv'])
        
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_default_data()

    # Identify products (assuming all columns are products)
    products = df.columns.tolist()
    
    # --- COMPUTATION ---
    with st.spinner('Running Evolutionary Solver... finding the optimal price anchors...'):
        # 1. Baseline
        sep_rev, sep_prices = calculate_baseline_separate(df, products)
        
        # 2. Optimization
        opt_params, mixed_rev = run_evolutionary_optimization(df, products)
        
        # 3. Process Results
        customer_df = generate_customer_details(df, products, opt_params)
        total_surplus = customer_df['Surplus'].sum()
        
        # Derived Stats
        uplift_pct = ((mixed_rev - sep_rev) / sep_rev) * 100
        bundle_price_opt = opt_params[len(products)]
        indiv_sum_opt = np.sum(opt_params[:len(products)])
        discount_pct = ((indiv_sum_opt - bundle_price_opt) / indiv_sum_opt) * 100

    # --- SECTION 1: TOP METRICS ---
    st.markdown("### 📊 Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style='margin:0; color:#64748b; font-size:0.9rem; text-transform:uppercase;'>Total Revenue</h4>
            <h2 style='margin:0; font-size:2.5rem; color:#1e293b;'>₹{mixed_rev:,.0f}</h2>
            <p style='color:#16a34a; font-weight:bold; margin-top:5px;'>▲ {uplift_pct:.1f}% vs Separate Pricing</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="surplus-card">
            <h4 style='margin:0; color:#64748b; font-size:0.9rem; text-transform:uppercase;'>Total Consumer Surplus</h4>
            <h2 style='margin:0; font-size:2.5rem; color:#1e293b;'>₹{total_surplus:,.0f}</h2>
            <p style='color:#64748b; margin-top:5px;'>Value retained by customers</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Spacer

    # --- SECTION 2 & 3: SPLIT VIEW ---
    col_left, col_right = st.columns([1, 2])

    # LEFT: AI RECOMMENDATIONS
    with col_left:
        st.subheader("🤖 AI Recommendations")
        
        # Dynamic Insight Generation
        anchor_insight = "high" if discount_pct > 15 else "moderate"
        
        st.markdown(f"""
        <div class="insight-box">
            <strong>1. The "Anchor Price" Strategy</strong><br>
            Individual prices have been set {anchor_insight} to act as psychological anchors. 
            The calculated Bundle Price of <strong>₹{bundle_price_opt:,.0f}</strong> offers a 
            <strong>{discount_pct:.1f}% discount</strong> compared to sum-of-parts, maximizing conversion.
        </div>
        
        <div class="insight-box" style="border-left-color: #f97316; background-color: #fff7ed;">
            <strong>2. Cross-Sell Opportunity</strong><br>
            Analysis shows {len(customer_df[customer_df['Decision']=='Bundle'])} customers chose the full bundle. 
            For the {len(customer_df[customer_df['Decision']=='Individual'])} customers buying individually, 
            market a "Mini-Bundle" of the top 2 rejected items to capture mid-tier surplus.
        </div>
        
        <div class="insight-box" style="border-left-color: #a855f7; background-color: #f3e8ff;">
            <strong>3. Competitor Benchmarking</strong><br>
            Your optimal bundle price effectively prices each item at 
            <strong>₹{(bundle_price_opt/len(products)):,.0f}</strong> on average. 
            Highlight this "effective unit price" in marketing to undercut single-product competitors.
        </div>
        """, unsafe_allow_html=True)

    # RIGHT: CUSTOMER TABLE
    with col_right:
        st.subheader("👥 Customer Decisions")
        
        # Styling the dataframe
        st.dataframe(
            customer_df,
            column_config={
                "Customer ID": st.column_config.NumberColumn(format="#%d"),
                "Revenue": st.column_config.NumberColumn(format="₹%d"),
                "Surplus": st.column_config.ProgressColumn(
                    format="₹%d",
                    min_value=0,
                    max_value=int(customer_df['Surplus'].max()),
                ),
                "Decision": st.column_config.TextColumn(),
            },
            use_container_width=True,
            height=350,
            hide_index=True
        )

    # --- SECTION 4: PRICING CONFIGURATION ---
    st.write("---")
    st.subheader("🏷️ Optimal Pricing Configuration")
    
    # Create columns for cards (Products + Bundle)
    cols = st.columns(len(products) + 1)
    
    # Individual Cards
    for i, prod in enumerate(products):
        price = opt_params[i]
        clean_name = prod.replace("Samsung_", "").replace("_", " ")
        with cols[i]:
            st.markdown(f"""
            <div class="price-card">
                <div style='font-size:0.8rem; font-weight:bold; color:#64748b; text-transform:uppercase; height:30px;'>{clean_name}</div>
                <div style='font-size:1.2rem; font-weight:bold; color:#1e293b; margin: 5px 0;'>₹{price:,.0f}</div>
                <div style='font-size:0.7rem; color:#ef4444;'>Anchor Price</div>
            </div>
            """, unsafe_allow_html=True)

    # Bundle Card
    with cols[-1]:
        st.markdown(f"""
        <div class="bundle-card">
            <div style='font-size:0.8rem; font-weight:bold; opacity:0.9; text-transform:uppercase;'>Full Bundle</div>
            <div style='font-size:1.4rem; font-weight:bold; margin: 5px 0;'>₹{bundle_price_opt:,.0f}</div>
            <div style='font-size:0.7rem; opacity:0.8; text-decoration: line-through;'>Sum: ₹{indiv_sum_opt:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()