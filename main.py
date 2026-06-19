from preprocess import encode_text
import sys
import joblib

model_file = "spam_detector.joblib"
joblib_model = joblib.load(model_file)

def predict_message(message):
    encoded = encode_text(message)
    score = float(joblib_model.predict(encoded, verbose=0)[0][0])
    label_name = "spam" if score >= 0.5 else "nie spam"
    return label_name, score


if len(sys.argv) > 1:
    message = " ".join(sys.argv[1:]).strip()
else:
    message = input("podaj wiadomosc do sprawdzenia: ").strip()

if message:
    result, score = predict_message(message)
    print(f"wynik: {result} (prawdopodobieństwo spamu: {score:.6f})")
