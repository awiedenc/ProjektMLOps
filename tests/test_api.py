from api.services import _rename_breast_columns

test_payload = {
    "concave_points_mean": 0.1471,
    "concave_points_se": 0.01587,
    "concave_points_worst": 0.2654,
    "radius_mean": 17.99
}

renamed = _rename_breast_columns(test_payload)
print("=== Mapowanie API ===")
print("Original:", test_payload)
print("Renamed:", renamed)
print("Sprawdzanie mapowania:",
      all("concave points" in key for key in renamed if "concave points" in key))