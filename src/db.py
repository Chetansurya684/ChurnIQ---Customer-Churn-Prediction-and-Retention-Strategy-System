import os
import pandas as pd
import mysql.connector


def cfg():
    return {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "churn_db"),
        "auth_plugin": "caching_sha2_password",
        "use_pure": True,
        "connection_timeout": 5
    }


def conn():
    return mysql.connector.connect(**cfg())


def test():
    try:
        c = conn()
        ok = c.is_connected()
        c.close()

        if ok:
            return True, "MySQL connection successful."

        return False, "MySQL connection failed."

    except Exception as e:
        return False, f"MySQL Error: {e}"


def init():
    c = conn()
    q = c.cursor()

    q.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id VARCHAR(50) PRIMARY KEY,
            tenure INT,
            monthly_charges DECIMAL(10,2),
            contract VARCHAR(50),
            internet_service VARCHAR(50),
            tech_support VARCHAR(20),
            payment_method VARCHAR(100),
            senior_citizen TINYINT,
            partner VARCHAR(10),
            paperless_billing VARCHAR(10),
            actual_churn TINYINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    q.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(50),
            churn_probability DECIMAL(8,5),
            risk_level VARCHAR(20),
            model_name VARCHAR(100),
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id)
                REFERENCES customers(customer_id)
                ON DELETE CASCADE
        )
    """)

    q.execute("""
        CREATE TABLE IF NOT EXISTS retention_actions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(50),
            risk_level VARCHAR(20),
            recommended_action TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id)
                REFERENCES customers(customer_id)
                ON DELETE CASCADE
        )
    """)

    c.commit()
    q.close()
    c.close()


def save_customers(df):
    c = conn()
    q = c.cursor()

    sql = """
        INSERT INTO customers (
            customer_id,
            tenure,
            monthly_charges,
            contract,
            internet_service,
            tech_support,
            payment_method,
            senior_citizen,
            partner,
            paperless_billing,
            actual_churn,
            created_at
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())

        ON DUPLICATE KEY UPDATE
            tenure = VALUES(tenure),
            monthly_charges = VALUES(monthly_charges),
            contract = VALUES(contract),
            internet_service = VALUES(internet_service),
            tech_support = VALUES(tech_support),
            payment_method = VALUES(payment_method),
            senior_citizen = VALUES(senior_citizen),
            partner = VALUES(partner),
            paperless_billing = VALUES(paperless_billing),
            actual_churn = VALUES(actual_churn)
    """

    for _, r in df.iterrows():
        q.execute(
            sql,
            (
                r.customer_id,
                int(r.tenure),
                float(r.monthly_charges),
                r.contract,
                r.internet_service,
                r.tech_support,
                r.payment_method,
                int(r.senior_citizen),
                r.partner,
                r.paperless_billing,
                int(r.churn)
            )
        )

    c.commit()
    q.close()
    c.close()


def save_prediction(cid, p, risk, model, actions):
    c = conn()
    q = c.cursor()

    q.execute(
        """
        INSERT INTO predictions
        (customer_id, churn_probability, risk_level, model_name)
        VALUES (%s, %s, %s, %s)
        """,
        (cid, p, risk, model)
    )

    for action in actions:
        q.execute(
            """
            INSERT INTO retention_actions
            (customer_id, risk_level, recommended_action)
            VALUES (%s, %s, %s)
            """,
            (cid, risk, action)
        )

    c.commit()
    q.close()
    c.close()


def history():
    c = conn()

    d = pd.read_sql(
        """
        SELECT
            customer_id,
            churn_probability,
            risk_level,
            model_name,
            predicted_at
        FROM predictions
        ORDER BY predicted_at DESC
        LIMIT 100
        """,
        c
    )

    c.close()
    return d