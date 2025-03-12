import streamlit as st
import requests
from streamlit_lottie import st_lottie

# --------------------------------------------About & Credits -----------------------------------------------------

st.subheader('About :question:')

def load_lottieurl(url: str):
    import requests  # Ensure requests module is imported
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_about = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sy6jjyct.json")

# +
col1, col2 = st.columns([6, 3])

with col1:
    st.write("""
        The purpose of this platform is to assist financial institutions and lenders in assessing loan repayment risks.
        \nThis application provides key functionalities:
        \n**📊 Exploratory Data Analysis**: Gain insights from loan applicant data through detailed EDA.
        \n**🔍 Single Prediction**: Assess an individual's loan repayment risk based on their financial and behavioral profile.
        \n**📂 Batch Prediction**: Perform risk assessment for multiple applicants using a dataset file, displaying outcomes for all records.
        \n**📌 About | Credits**: Share your feedback and suggestions for future enhancements!
    """)

with col2:
    st_lottie(
        lottie_about,
        height=340,
        width=None,
        quality="high",
        key=None,
    )
# -

"---"

st.subheader('Documentation :memo:')
st.write("""The Website User Manual is available here.""")
st.write("""[User Manual](https://drive.google.com/file/d/19jXwrDRwvzRH8blj8xpRkgY_afelRTxU/view?usp=share_link)""")

"---"

st.subheader('Credits :star2: :computer:')
st.write("""
    \nThis platform is a prototype and mockup design developed as part of the MoneyLion's Take Home Assessment for the `Loan Repayment Challenge`.
    \nThe website was created by Kong Jing Yuaan to demonstrate data science and machine learning capabilities in loan risk prediction.
""")

"---"

st.subheader(":mailbox: Get In Touch With Me!")
st.write("""\nIf you have any feedback, don't hesitate to fill in this form.""")

contact_form = """
<form action="https://formsubmit.co/kjyuaan8@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your feedback here"></textarea>
     <button type="submit">Send</button>
</form>
"""
st.markdown(contact_form, unsafe_allow_html=True)

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")
