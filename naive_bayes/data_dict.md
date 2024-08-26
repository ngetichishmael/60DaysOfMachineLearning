- A loan data dataset from [Kaggle](https://www.kaggle.com/datasets/itssuru/loan-data)
- Classify and predict whether or not the borrower paid back their loan in full.

| Variable            | Explanation                                                                                                      |
|---------------------|------------------------------------------------------------------------------------------------------------------|
| credit_policy       | 1 if the customer meets the credit underwriting criteria; 0 otherwise.                                            |
| purpose             | The purpose of the loan.                                                                                          |
| int_rate            | The interest rate of the loan (more risky borrowers are assigned higher interest rates).                          |
| installment         | The monthly installments owed by the borrower if the loan is funded.                                              |
| log_annual_inc      | The natural log of the self-reported annual income of the borrower.                                               |
| dti                 | The debt-to-income ratio of the borrower (amount of debt divided by annual income).                               |
| fico                | The FICO credit score of the borrower.                                                                            |
| days_with_cr_line   | The number of days the borrower has had a credit line.                                                            |
| revol_bal           | The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).                     |
| revol_util          | The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available). |
| inq_last_6mths      | The borrower's number of inquiries by creditors in the last 6 months.                                             |
| delinq_2yrs         | The number of times the borrower had been 30+ days past due on a payment in the past 2 years.                     |
| pub_rec             | The borrower's number of derogatory public records.                                                               |
| not_fully_paid      | 1 if the loan is not fully paid; 0 otherwise.                                                                     |
