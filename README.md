<div align="center" id="top"> 
  <img src="./.github/app.gif" alt="Customer Churn " />

  &#xa0;

  <!-- <a href="https://customer_churn.netlify.app">Demo</a> -->
</div>

<h1 align="center">Customer Churn Prediction</h1>

<!-- Status -->

<!-- <h4 align="center"> 
	🚧  Customer_Churn 🚀 Under construction...  🚧
</h4> 

<hr> -->

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#sparkles-features">Features</a> &#xa0; | &#xa0;
  <a href="#rocket-technologies">Technologies</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Starting</a> &#xa0; | &#xa0;
  <a href="#memo-license">License</a> &#xa0; | &#xa0;
  <a href="https://github.com/{{YOUR_GITHUB_USERNAME}}" target="_blank">Author</a>
</p>

<br>

## :dart: About ##

This project is a Customer Churn Prediction application built using Streamlit and PyTorch. It allows users to predict whether a customer is likely to churn (leave the service) based on various customer attributes, such as credit score, age, balance, and more.

## :sparkles: Features ##

The following features are used to predict customer churn in this model:

- **CreditScore**: The credit score of the customer. Higher scores indicate a better credit history.
- **Age**: The age of the customer. Age is an important factor in predicting churn as it correlates with life stages and financial stability.
- **Tenure**: The number of years the customer has been with the company. Long tenure can indicate loyalty and reduce the likelihood of churn.
- **Balance**: The current balance of the customer’s account. Customers with low balances may be more likely to churn.
- **NumOfProducts**: The number of products the customer holds with the company. Having more products may reduce churn likelihood.
- **HasCrCard**: Whether the customer has a credit card with the company (1 for Yes, 0 for No). Having a credit card could reduce churn chances.
- **IsActiveMember**: Indicates whether the customer is an active member (1 for Yes, 0 for No). Active members are less likely to churn.
- **EstimatedSalary**: The estimated annual salary of the customer. This can be an indicator of financial stability and churn likelihood.
- **Geography**: The geographical location of the customer. This feature includes the following values:
  - **France**
  - **Germany**
  - **Spain**
  
  These are encoded into numerical features using One-Hot Encoding.
  
- **Gender**: The gender of the customer (Male or Female). This feature is encoded using Label Encoding.

## :rocket: Technologies ##

The following tools were used in this project:

- [Python](https://docs.python.org/3/)
- [Pytorch](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Streamlit](https://docs.streamlit.io/)
- [Pickle](https://docs.python.org/3/library/pickle.html)

## :white_check_mark: Requirements ##

Before starting :checkered_flag:, you need to have [Git](https://git-scm.com) and [Python](https://www.python.org) installed.

## :checkered_flag: Starting ##

```bash
# Clone this project
$ git clone https://github.com/Dhairyakhania/Customer-Churn-Prediction.git

# Access
$ cd Customer-Churn-Prediction

# Install dependencies
$ pip install streamlit torch pandas numpy scikit-learn pickle


# Run the project
$ streamlit run app.py

# The server will initialize in the <http://localhost:3000>
```

## :memo: License ##

This project is under license from MIT. For more details, see the [LICENSE](LICENSE.md) file.


Made with :heart: by <a href="https://github.com/{{YOUR_GITHUB_USERNAME}}" target="_blank">Dhairya Khania</a>

&#xa0;

<a href="#top">Back to top</a>
