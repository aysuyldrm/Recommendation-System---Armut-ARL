🔍 Business Problem

Armut connects service providers with customers seeking services such as:

Cleaning

Renovation

Transportation

Using historical transaction data, the objective is to:

Discover meaningful service associations

Recommend additional services to users based on past behavior

📂 Dataset Overview

The dataset contains historical service purchase data with the following fields:

UserId: Unique customer ID

ServiceId: Anonymized service ID

CategoryId: Anonymized service category

CreateDate: Date and time of service purchase

⚠️ Since the same ServiceId may represent different services across categories,
a composite service identifier is created:

Service = ServiceId_CategoryId

🛒 Basket (Transaction) Definition

There is no predefined basket or invoice structure in the dataset.

To apply Association Rule Learning:

A basket is defined as all services a user purchases within the same month

Each basket is uniquely identified as:

BasketID = UserId_Year-Month


This approach captures monthly service consumption behavior per user.

🧠 Methodology
Data Preparation

Combine ServiceId and CategoryId into a single service identifier

Convert timestamps to Year–Month format

Create basket-based one-hot encoded transaction matrix

Association Rule Learning

Apply Apriori algorithm with minimum support threshold

Generate association rules using:

Support

Confidence

Lift

Rank rules by lift to identify strongest service relationships

🤖 Recommendation Logic

A custom function arl_recommender():

Takes a target service as input

Finds rules where the service appears in the antecedent

Returns the most strongly associated services (by lift)

Removes duplicate recommendations

Example:
arl_recommender(rules, "2_0", 4)


Sample Output:

['38_4', '15_1', '25_0', '13_11']

📊 Output

Personalized service recommendations

Data-driven cross-selling insights

Interpretable rules suitable for business use

🛠️ Technologies Used

Python

Pandas

mlxtend (Apriori, association_rules)

Data preprocessing & feature engineering

Market Basket Analysis
