/*
============================================================
HOME CREDIT DEFAULT RISK
CUSTOMER-LEVEL FEATURE ENGINEERING
============================================================

Database: PostgreSQL
Customer Key: SK_ID_CURR
Target: TARGET

Source tables:
    - application_train
    - credit_card_balance
    - bureau
    - bureau_balance
    - installments_payments
    - previous_application

Goal:
    Transform historical transactional data into
    customer-level features.

Final granularity:
    1 row = 1 customer

============================================================
*/


WITH

/* =========================================================
   1. CREDIT CARD - CARD LEVEL
   ========================================================= */

card_features AS (

    SELECT
        ccb."SK_ID_CURR",
        ccb."SK_ID_PREV",

        COUNT(*) AS card_month_count,

        MAX(
            ccb."AMT_CREDIT_LIMIT_ACTUAL"
        ) AS card_max_credit_limit,

        AVG(
            ccb."AMT_DRAWINGS_ATM_CURRENT"
        ) AS card_avg_atm_drawings,

        AVG(
            ccb."AMT_DRAWINGS_CURRENT"
        ) AS card_avg_drawings,

        SUM(
            ccb."AMT_DRAWINGS_CURRENT"
        ) AS card_total_drawings,

        AVG(
            ccb."AMT_INST_MIN_REGULARITY"
        ) AS card_avg_min_regularity,

        AVG(
            ccb."AMT_PAYMENT_TOTAL_CURRENT"
        ) AS card_avg_payment,

        AVG(
            ccb."AMT_TOTAL_RECEIVABLE"
        ) AS card_avg_total_receivable,

        MAX(
            ccb."SK_DPD"
        ) AS card_max_dpd,

        MAX(
            ccb."SK_DPD_DEF"
        ) AS card_max_dpd_def,

        COUNT(*) FILTER (
            WHERE ccb."SK_DPD" > 0
        ) AS card_dpd_month_count,

        COUNT(*) FILTER (
            WHERE ccb."SK_DPD_DEF" > 0
        ) AS card_dpd_def_month_count

    FROM credit_card_balance ccb

    GROUP BY
        ccb."SK_ID_CURR",
        ccb."SK_ID_PREV"
),


/* =========================================================
   2. CREDIT CARD - CUSTOMER LEVEL
   ========================================================= */

credit_card_customer AS (

    SELECT
        "SK_ID_CURR",

        /* Number of credit cards */
        COUNT(*) AS card_count,

        /* Total observed months */
        SUM(card_month_count) AS total_months_balance,

        /* Total credit limit across cards */
        SUM(card_max_credit_limit) AS total_credit_limit,

        /* Maximum credit limit of any card */
        MAX(card_max_credit_limit) AS max_credit_limit,

        /* Average behavior across cards */
        AVG(card_avg_atm_drawings)
            AS avg_atm_drawings,

        AVG(card_avg_drawings)
            AS avg_drawings,

        /* Total spending */
        SUM(card_total_drawings)
            AS total_drawings,

        AVG(card_avg_min_regularity)
            AS avg_min_regularity,

        AVG(card_avg_payment)
            AS avg_payment,

        AVG(card_avg_total_receivable)
            AS avg_total_receivable,

        /* Worst delinquency */
        MAX(card_max_dpd)
            AS max_dpd,

        MAX(card_max_dpd_def)
            AS max_dpd_def,

        /* Number of months with delinquency */
        SUM(card_dpd_month_count)
            AS dpd_month_count,

        SUM(card_dpd_def_month_count)
            AS dpd_def_month_count,

        /* Delinquency frequency */
        SUM(card_dpd_month_count)::numeric
        /
        NULLIF(SUM(card_month_count), 0)
            AS dpd_frequency,

        /* Default delinquency frequency */
        SUM(card_dpd_def_month_count)::numeric
        /
        NULLIF(SUM(card_month_count), 0)
            AS dpd_def_frequency

    FROM card_features

    GROUP BY
        "SK_ID_CURR"
),


/* =========================================================
   3. BUREAU BALANCE - CREDIT LEVEL
   ========================================================= */

bureau_balance_features AS (

    SELECT
        bb."SK_ID_BUREAU",

        COUNT(*) AS observation_months,

        COUNT(*) FILTER (
            WHERE bb."STATUS" IN ('1', '2', '3', '4', '5')
        ) AS overdue_months,

        MAX(
            CASE
                WHEN bb."STATUS" IN ('1', '2', '3', '4', '5')
                    THEN CAST(bb."STATUS" AS INTEGER)
                ELSE 0
            END
        ) AS max_status

    FROM bureau_balance bb

    GROUP BY
        bb."SK_ID_BUREAU"
),


/* =========================================================
   4. BUREAU - CUSTOMER LEVEL
   ========================================================= */

bureau_features AS (

    SELECT
        b."SK_ID_CURR",

        /* Credit count */
        COUNT(*) AS bureau_credit_count,

        COUNT(*) FILTER (
            WHERE b."CREDIT_ACTIVE" = 'Active'
        ) AS active_credit_count,

        COUNT(*) FILTER (
            WHERE b."CREDIT_ACTIVE" = 'Closed'
        ) AS closed_credit_count,

        /* Credit amounts */
        SUM(b."AMT_CREDIT_SUM")
            AS total_credit_sum,

        SUM(b."AMT_CREDIT_SUM_DEBT")
            AS total_debt,

        MAX(b."AMT_CREDIT_SUM")
            AS max_credit_sum,

        AVG(b."AMT_CREDIT_SUM")
            AS avg_credit_sum,

        /* Overdue information */
        MAX(b."CREDIT_DAY_OVERDUE")
            AS max_credit_day_overdue,

        SUM(b."CREDIT_DAY_OVERDUE")
            AS total_credit_day_overdue,

        /* Bureau balance history */
        COALESCE(
            SUM(bb.observation_months),
            0
        ) AS total_observation_months,

        COALESCE(
            SUM(bb.overdue_months),
            0
        ) AS total_overdue_months,

        COALESCE(
            SUM(bb.overdue_months),
            0
        )::numeric
        /
        NULLIF(
            SUM(bb.observation_months),
            0
        ) AS overdue_frequency,

        MAX(bb.max_status)
            AS max_status

    FROM bureau b

    LEFT JOIN bureau_balance_features bb
        ON b."SK_ID_BUREAU"
        = bb."SK_ID_BUREAU"

    GROUP BY
        b."SK_ID_CURR"
),


/* =========================================================
   5. INSTALLMENT PAYMENTS - CUSTOMER LEVEL
   ========================================================= */

installment_features AS (

    SELECT
        ip."SK_ID_CURR",

        /* Number of previous credits */
        COUNT(
            DISTINCT ip."SK_ID_PREV"
        ) AS installment_credit_count,

        /* Number of installments */
        COUNT(*) AS total_installment_count,

        /* Installment amounts */
        SUM(ip."AMT_INSTALMENT")
            AS total_installment_amount,

        SUM(ip."AMT_PAYMENT")
            AS total_payment_amount,

        /* Payment ratio */
        SUM(ip."AMT_PAYMENT")
        /
        NULLIF(
            SUM(ip."AMT_INSTALMENT"),
            0
        ) AS payment_ratio,

        /* Late payment days */
        SUM(
            GREATEST(
                ip."DAYS_ENTRY_PAYMENT"
                - ip."DAYS_INSTALMENT",
                0
            )
        ) AS total_late_days,

        MAX(
            GREATEST(
                ip."DAYS_ENTRY_PAYMENT"
                - ip."DAYS_INSTALMENT",
                0
            )
        ) AS max_late_days,

        AVG(
            GREATEST(
                ip."DAYS_ENTRY_PAYMENT"
                - ip."DAYS_INSTALMENT",
                0
            )
        ) AS avg_late_days,

        /* Late installments */
        COUNT(*) FILTER (
            WHERE
                ip."DAYS_ENTRY_PAYMENT"
                > ip."DAYS_INSTALMENT"
        ) AS late_installment_count,

        COUNT(*) FILTER (
            WHERE
                ip."DAYS_ENTRY_PAYMENT"
                > ip."DAYS_INSTALMENT"
        )::numeric
        /
        NULLIF(COUNT(*), 0)
            AS late_payment_frequency,

        /* Underpayment */
        COUNT(*) FILTER (
            WHERE
                ip."AMT_PAYMENT"
                < ip."AMT_INSTALMENT"
        ) AS underpaid_installment_count,

        /* Fully paid */
        COUNT(*) FILTER (
            WHERE
                ip."AMT_PAYMENT"
                >= ip."AMT_INSTALMENT"
        ) AS fully_paid_installment_count,

        /* Payment gap */
        SUM(
            ip."AMT_INSTALMENT"
            - ip."AMT_PAYMENT"
        ) AS total_payment_gap

    FROM installments_payments ip

    GROUP BY
        ip."SK_ID_CURR"
),


/* =========================================================
   6. PREVIOUS APPLICATIONS - CUSTOMER LEVEL
   ========================================================= */

previous_application_features AS (

    SELECT
        pa."SK_ID_CURR",

        /* Application history */
        COUNT(*) AS previous_application_count,

        COUNT(*) FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Approved'
        ) AS approved_count,

        COUNT(*) FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Refused'
        ) AS refused_count,

        /* Approval / refusal rates */
        COUNT(*) FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Approved'
        )::numeric
        /
        NULLIF(COUNT(*), 0)
            AS approval_rate,

        COUNT(*) FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Refused'
        )::numeric
        /
        NULLIF(COUNT(*), 0)
            AS refusal_rate,

        /* Approved credit */
        MAX(pa."AMT_CREDIT") FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Approved'
        ) AS max_approved_credit,

        AVG(pa."AMT_CREDIT") FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Approved'
        ) AS avg_approved_credit,

        /* Refused credit */
        MIN(pa."AMT_CREDIT") FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Refused'
        ) AS min_refused_credit,

        /* Approved annuity */
        MAX(pa."AMT_ANNUITY") FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Approved'
        ) AS max_approved_annuity,

        AVG(pa."AMT_ANNUITY") FILTER (
            WHERE pa."NAME_CONTRACT_STATUS" = 'Approved'
        ) AS avg_approved_annuity,

        /* Recency of previous application */
        MAX(pa."DAYS_DECISION")
            AS latest_previous_application,

        /* Average requested credit */
        AVG(pa."AMT_CREDIT")
            AS avg_previous_credit

    FROM previous_application pa

    GROUP BY
        pa."SK_ID_CURR"
)


/* =========================================================
   7. FINAL CUSTOMER-LEVEL DATASET
   ========================================================= */

SELECT

    t.*,

    /* =====================================================
       CREDIT CARD FEATURES
       ===================================================== */

    cc.card_count,
    cc.total_months_balance,
    cc.total_credit_limit,
    cc.max_credit_limit,
    cc.avg_atm_drawings,
    cc.avg_drawings,
    cc.total_drawings,
    cc.avg_min_regularity,
    cc.avg_payment,
    cc.avg_total_receivable,
    cc.max_dpd,
    cc.max_dpd_def,
    cc.dpd_month_count,
    cc.dpd_def_month_count,
    cc.dpd_frequency,
    cc.dpd_def_frequency,


    /* =====================================================
       BUREAU FEATURES
       ===================================================== */

    b.bureau_credit_count,
    b.active_credit_count,
    b.closed_credit_count,
    b.total_credit_sum,
    b.total_debt,
    b.max_credit_sum,
    b.avg_credit_sum,
    b.max_credit_day_overdue,
    b.total_credit_day_overdue,
    b.total_observation_months,
    b.total_overdue_months,
    b.overdue_frequency,
    b.max_status,


    /* =====================================================
       INSTALLMENT FEATURES
       ===================================================== */

    i.installment_credit_count,
    i.total_installment_count,
    i.total_installment_amount,
    i.total_payment_amount,
    i.payment_ratio,
    i.total_late_days,
    i.max_late_days,
    i.avg_late_days,
    i.late_installment_count,
    i.late_payment_frequency,
    i.underpaid_installment_count,
    i.fully_paid_installment_count,
    i.total_payment_gap,


    /* =====================================================
       PREVIOUS APPLICATION FEATURES
       ===================================================== */

    p.previous_application_count,
    p.approved_count,
    p.refused_count,
    p.approval_rate,
    p.refusal_rate,
    p.max_approved_credit,
    p.avg_approved_credit,
    p.min_refused_credit,
    p.max_approved_annuity,
    p.avg_approved_annuity,
    p.latest_previous_application,
    p.avg_previous_credit


FROM application_train t


/* =========================================================
   FINAL JOINS
   ========================================================= */

LEFT JOIN credit_card_customer cc
    ON t.sk_id_curr = cc."SK_ID_CURR"

LEFT JOIN bureau_features b
    ON t.sk_id_curr = b."SK_ID_CURR"

LEFT JOIN installment_features i
    ON t.sk_id_curr = i."SK_ID_CURR"

LEFT JOIN previous_application_features p
    ON t.sk_id_curr = p."SK_ID_CURR";
