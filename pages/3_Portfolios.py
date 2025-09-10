import streamlit as st
import pandas as pd
from src.storage import list_portfolios, create_portfolio, delete_portfolio, remove_item

st.set_page_config(page_title="Portfolios", page_icon="🗂️")
st.title("🗂️ Portfolios")

with st.expander("Create Portfolio"):
    name = st.text_input("Portfolio Name", value="My Portfolio")
    if st.button("Create"):
        if name.strip():
            p = create_portfolio(name.strip())
            st.success(f"Created portfolio '{p['name']}'")
        else:
            st.warning("Please enter a name.")

portfolios = list_portfolios()
if not portfolios:
    st.info("No portfolios yet. Create one above or save from the Tuning page.")
else:
    for p in portfolios:
        with st.expander(f"📁 {p['name']} — {len(p.get('items',[]))} items", expanded=False):
            items = p.get("items", [])
            if items:
                df = pd.DataFrame([{"symbol": it["symbol"], "model": it["model"], **it["params"]} for it in items])
                st.dataframe(df, use_container_width=True)
                cols = st.columns(3)
                with cols[0]:
                    sym = st.text_input(f"Remove symbol from '{p['name']}'", key=f"rm_{p['id']}")
                with cols[1]:
                    if st.button("Remove", key=f"btn_rm_{p['id']}"):
                        if sym.strip():
                            ok = remove_item(p["id"], sym.strip().upper(), "atr_breakout")
                            st.success("Removed." if ok else "Not found.")
                with cols[2]:
                    if st.button("Delete Portfolio", key=f"del_{p['id']}"):
                        if delete_portfolio(p["id"]):
                            st.success("Portfolio deleted. Refresh to see changes.")
                        else:
                            st.error("Failed to delete.")
            else:
                st.write("_No items yet._")
