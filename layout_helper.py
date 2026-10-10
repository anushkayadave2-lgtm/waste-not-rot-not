import streamlit as st

def apply_custom_theme():
    """Applies clean white theme with soft green accents and light grey sections."""
    st.markdown(
        """
        <style>
        /* Main background & font styling */
        .stApp {
            background-color: #FFFFFF;
            color: #1E293B;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }
        
        [data-testid="stSidebar"] .stMarkdown h1, 
        [data-testid="stSidebar"] .stMarkdown h2, 
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #166534;
        }

        /* Status Badge in Sidebar */
        .status-badge-active {
            background-color: #DCFCE7;
            color: #15803D;
            border: 1px solid #86EFAC;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 1rem;
        }
        
        .status-badge-inactive {
            background-color: #FEF3C7;
            color: #B45309;
            border: 1px solid #FDE68A;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 1rem;
        }

        /* Light Grey Card Sections */
        .section-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .section-card-green {
            background-color: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }

        /* Metric Cards */
        .metric-card-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .metric-card-box-na {
            background-color: #F8FAFC;
            border: 1px dashed #CBD5E1;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: none;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .metric-card-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #15803D;
            margin-top: 4px;
        }

        .metric-card-val-na {
            font-size: 1.6rem;
            font-weight: 600;
            color: #94A3B8;
            margin-top: 4px;
        }

        .metric-card-val-alert {
            font-size: 1.8rem;
            font-weight: 700;
            color: #DC2626;
            margin-top: 4px;
        }

        .metric-card-lbl {
            font-size: 0.8rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-card-sub {
            font-size: 0.78rem;
            color: #64748B;
            margin-top: 4px;
        }

        /* Alert row card styling */
        .alert-card-item {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }

        /* Pill badges */
        .status-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 14px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        /* Decision Status Badges */
        .decision-badge-approved {
            background-color: #DCFCE7;
            color: #15803D;
            border: 1px solid #86EFAC;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .decision-badge-rejected {
            background-color: #FEE2E2;
            color: #DC2626;
            border: 1px solid #FECACA;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .decision-badge-modified {
            background-color: #DBEAFE;
            color: #1D4ED8;
            border: 1px solid #93C5FD;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .decision-badge-pending {
            background-color: #FEF3C7;
            color: #B45309;
            border: 1px solid #FDE68A;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        /* Recommendation card container */
        .rec-card-box {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        /* Streamlit Buttons */
        .stButton>button {
            background-color: #16A34A;
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #15803D;
            color: white;
        }

        /* Headers */
        h1, h2, h3 {
            color: #0F172A;
            font-weight: 700;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

def setup_sidebar():
    """Renders the standard sidebar with leaf icon, store title, status indicator, and navigation."""
    with st.sidebar:
        st.markdown("### 🍃 Zero-Waste Grocery")
        st.caption("Store Inventory & Expiry Assistant")
        st.markdown("---")

        # Status Indicator
        is_uploaded = st.session_state.get("data_uploaded", False)
        is_demo = st.session_state.get("is_demo", False)
        has_df = st.session_state.get("df") is not None and not getattr(st.session_state.get("df"), "empty", True)

        if is_uploaded and not is_demo and has_df:
            st.markdown(
                '<div class="status-badge-active">🟢 Store Data Uploaded</div>',
                unsafe_allow_html=True,
            )
            file_name = st.session_state.get("uploaded_filename", "Custom File")
            st.caption(f"📁 Active File: `{file_name}`")
        elif is_demo and has_df:
            st.markdown(
                '<div class="status-badge-inactive">🟡 Demo Mode Active</div>',
                unsafe_allow_html=True,
            )
            st.caption("Viewing sample demo dataset.")
        else:
            st.markdown(
                '<div style="background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; padding: 8px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; margin-bottom: 0.75rem;">⚪ No Data Loaded</div>',
                unsafe_allow_html=True,
            )
            st.caption("Upload store spreadsheet to activate.")

        st.markdown("---")
        
        # Reset Data Action
        if st.button("🔄 Clear / Reset Data", use_container_width=True):
            st.session_state["data_uploaded"] = False
            st.session_state["is_demo"] = False
            st.session_state["uploaded_filename"] = None
            st.session_state["decisions"] = {}
            st.session_state["confirming_id"] = None
            st.session_state["confirming_action"] = None
            if "df" in st.session_state:
                st.session_state["df"] = None
            if "raw_df" in st.session_state:
                st.session_state["raw_df"] = None
            st.rerun()

        st.markdown("---")
        st.caption("💡 Non-technical, user-friendly grocery dashboard.")
