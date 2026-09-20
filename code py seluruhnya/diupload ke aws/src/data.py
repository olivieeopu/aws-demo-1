import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataIngestion:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):

        df = pd.read_csv(
            self.filepath
        )

        print(
            f"Dataset loaded: {df.shape}"
        )

        return df
    


class DataPreprocessor:

    def __init__(self):

        self.scaler = StandardScaler()

        self.encoder = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        )

        self.feature_names = None

    def preprocess(self, df):

        # DROP IDENTIFIER
        drop_cols = [
            'ID',
            'Customer_ID',
            'Name',
            'SSN',
            'Month',
            'Unnamed: 0'
        ]

        df = df.drop(columns=drop_cols, errors='ignore')

        # HANDLE INVALID VALUES
        df['Occupation'] = df['Occupation'].replace(
            '_______',
            np.nan
        )

        df['Credit_Mix'] = df['Credit_Mix'].replace(
            '_',
            np.nan
        )

        df['Payment_of_Min_Amount'] = (
            df['Payment_of_Min_Amount']
            .replace('NM', np.nan)
        )

        df['Payment_Behaviour'] = (
            df['Payment_Behaviour']
            .replace('!@9#%8', np.nan)
        )


        # CONVERT OBJECT NUMERIC
        numeric_string_cols = [
            'Age',
            'Annual_Income',
            'Num_of_Loan',
            'Num_of_Delayed_Payment',
            'Changed_Credit_Limit',
            'Outstanding_Debt',
            'Amount_invested_monthly',
            'Monthly_Balance'
        ]

        for col in numeric_string_cols:

            if col in df.columns:

                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace('_', '', regex=False)
                )

                df[col] = pd.to_numeric(
                    df[col],
                    errors='coerce'
                )

        # OUTLIER HANDLING
        rules = {
            'Age': (18, 100),
            'Num_Bank_Accounts': (0, 20),
            'Num_Credit_Card': (0, 20),
            'Interest_Rate': (0, 50),
            'Num_of_Loan': (0, 20),
            'Num_of_Delayed_Payment': (0, 100),
            'Num_Credit_Inquiries': (0, 50)
        }

        for col, (low, high) in rules.items():

            if col in df.columns:

                df.loc[
                    (df[col] < low) |
                    (df[col] > high),
                    col
                ] = np.nan

        df.loc[
            df['Delay_from_due_date'] < 0,
            'Delay_from_due_date'
        ] = np.nan

        # MISSING VALUE HANDLING

        df['Occupation'] = (
            df['Occupation']
            .fillna('Unknown')
        )

        df['Type_of_Loan'] = (
            df['Type_of_Loan']
            .fillna('Unknown')
        )

        df['Credit_Mix'] = (
            df['Credit_Mix']
            .fillna(df['Credit_Mix'].mode()[0])
        )

        df['Payment_of_Min_Amount'] = (
            df['Payment_of_Min_Amount']
            .fillna(df['Payment_of_Min_Amount'].mode()[0])
        )

        df['Payment_Behaviour'] = (
            df['Payment_Behaviour']
            .fillna(df['Payment_Behaviour'].mode()[0])
        )

        numeric_cols = df.select_dtypes(
            include=['int64', 'float64']
        ).columns

        for col in numeric_cols:

            df[col] = df[col].fillna(
                df[col].median()
            )


        # SKEWNESS HANDLING

        skewed_cols = [
            'Annual_Income',
            'Total_EMI_per_month',
            'Amount_invested_monthly',
            'Monthly_Balance',
            'Monthly_Inhand_Salary',
            'Outstanding_Debt'
        ]

        for col in skewed_cols:

            if col in df.columns:

                df[col] = np.log1p(df[col])


        # FEATURE ENGINEERING

        credit_history = (
            df['Credit_History_Age']
            .astype(str)
            .str.extract(
                r'(\d+)\s+Years\s+and\s+(\d+)\s+Months'
                )
        )

        years = pd.to_numeric(
            credit_history[0],
            errors='coerce'
        )

        months = pd.to_numeric(
            credit_history[1],
            errors='coerce'
        )

        df['Credit_History_Years']=(
            years + (months / 12)
        )

        df['Debt_to_Income_Ratio'] = np.where(
            df['Annual_Income'] > 0,
            df['Outstanding_Debt'] / df['Annual_Income'],
            0
        )

        df['EMI_Burden'] = (
            df['Total_EMI_per_month']
            /
            (df['Monthly_Inhand_Salary'] + 1)
        )

        df['Financial_Exposure'] = (
            df['Num_Bank_Accounts']
            +
            df['Num_Credit_Card']
            +
            df['Num_of_Loan']
        )

        df['Risk_Score'] = (
            0.5 * df['Delay_from_due_date']
            +
            0.3 * df['Num_of_Delayed_Payment']
            +
            0.2 * df['Num_Credit_Inquiries']
        )

        def classify_risk(row):

            if (
                row['Delay_from_due_date'] > 30 or
                row['Num_of_Delayed_Payment'] > 10 or
                row['Num_Credit_Inquiries'] > 8
            ):
                return 'High'
        
            elif (
                row['Delay_from_due_date'] > 10 or
                row['Num_of_Delayed_Payment'] > 4 or
                row['Num_Credit_Inquiries'] > 4
            ):
                return 'Medium'
        
            else:
                return 'Low'

        df['Risk_Level'] = df.apply(
            classify_risk,
            axis=1
        )

        risk_mapping = {
            'Low': 0,
            'Medium': 1,
            'High': 2
        }

        df['Risk_Level'] = (
            df['Risk_Level']
            .map(risk_mapping)
            )

        # TARGET MAPPING
     
        df['Credit_Score'] = df['Credit_Score'].map({
            'Poor': 0,
            'Standard': 1,
            'Good': 2
        })

        # Cek
        numeric_cols = df.select_dtypes(
            include=['int64', 'float64']
        ).columns

        for col in numeric_cols:
            df[col] = df[col].fillna(
            df[col].median()
            )

        # FEATURE SELECTION

        selected_features = [
            'Occupation',
            'Num_Bank_Accounts',
            'Num_Credit_Card',
            'Interest_Rate',
            'Num_of_Loan',
            'Delay_from_due_date',
            'Num_of_Delayed_Payment',
            'Changed_Credit_Limit',
            'Num_Credit_Inquiries',
            'Credit_Mix',
            'Outstanding_Debt',
            'Credit_History_Years',
            'Payment_of_Min_Amount',
            'Payment_Behaviour',
            'Debt_to_Income_Ratio',
            'Risk_Score',
            'Financial_Exposure',
            'Credit_Score'
        ]


        df = df[selected_features]

        print("\n===== AFTER FEATURE SELECTION =====")
        print(df.columns.tolist())
        print("\nTotal Columns:", len(df.columns))


        # ONE HOT ENCODING

        categorical_cols = [
            'Occupation',
            'Credit_Mix',
            'Payment_of_Min_Amount',
            'Payment_Behaviour'
        ]

        encoded = self.encoder.fit_transform(
            df[categorical_cols]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(
                categorical_cols
            ),
            index=df.index
        )

        df = pd.concat(
            [
                df.drop(columns=categorical_cols),
                encoded_df
            ],
            axis=1
        )

        
        # SPLIT

        X = df.drop(columns=['Credit_Score'])

        y = df['Credit_Score']

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        print(X.shape)
        print(X.columns.tolist())
            

        # SCALING
        X_train_scaled = self.scaler.fit_transform(
            X_train
        )

        X_test_scaled = self.scaler.transform(
            X_test
        )

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )