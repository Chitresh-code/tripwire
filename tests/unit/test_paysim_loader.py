from datetime import datetime

from src.ingestion.paysim_loader import load_paysim


def test_load_paysim_maps_to_canonical_schema(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,"
        "isFraud,isFlaggedFraud\n"
        "1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0\n"
        "2,TRANSFER,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0\n"
    )

    result = load_paysim(csv_path)

    assert list(result.columns) == [
        "transaction_id",
        "account_id",
        "recipient_id",
        "timestamp",
        "amount",
        "transaction_type",
        "is_fraud",
    ]
    assert result["account_id"].tolist() == ["C1231006815", "C840083671"]
    assert result["recipient_id"].tolist() == ["M1979787155", "C38997010"]
    assert result["amount"].tolist() == [9839.64, 181.0]
    assert result["transaction_type"].tolist() == ["PAYMENT", "TRANSFER"]
    assert result["is_fraud"].tolist() == [False, True]
    assert result["timestamp"].tolist() == [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2)]
