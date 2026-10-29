Muhammwd Ilyas Khan
# YOLO11 Object Detection - Full Stack App

End-to-end object detection: Roboflow dataset -> YOLO11 training -> FastAPI backend -> web frontend.
Works with images and videos.

## Structure
```
training/   download_dataset.py, train.py
backend/    FastAPI API (main.py), Dockerfile
frontend/   index.html, style.css, script.js
weights/    best.pt (your trained model)
```

## 1. Dataset (Roboflow)
```bash
pip install roboflow
export ROBOFLOW_API_KEY=your_key        # Windows PowerShell: $env:ROBOFLOW_API_KEY="your_key"
python training/download_dataset.py --workspace <workspace> --project <project> --version 1
```

## 2. Train
```bash
python training/train.py --data dataset/data.yaml --epochs 60
```
Best model is saved to `weights/best.pt`. Use a GPU (Google Colab is fine).

## 3. Run the backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API docs: http://127.0.0.1:8000/docs

## 4. Run the frontend
```bash
cd frontend
python -m http.server 5500
```
Open http://127.0.0.1:5500

## Run everything in Google Colab
Use `colab_run.ipynb`: it unpacks the project, loads `best.pt`, starts the server (API + web page on one port) and opens it inside the notebook.

## Signature identification (whose signature is it?)
After a signature is found, a second model (`weights/people_cls.pt`) can tell **whose** it is.
Without that file the app only detects signatures.

```
Image -> YOLO11 detector (finds the signature) -> crop -> YOLO11-cls (who signed?) -> name + confidence
```

1. Collect at least **8 different signatures per person** (different documents / years) in
   `raw_signatures/<Person_Name>/`. Use public figures or people who gave permission.
2. Open `colab_train_people.ipynb` in Google Colab and run it. It crops the signatures with
   the detector, splits them into train/val, trains the classifier and gives you `people_cls.pt`.
3. Put `people_cls.pt` in `weights/` next to `best.pt` and start the app. The page now shows
   the person's name, match percentage and the closest other matches.

Settings (environment variables): `PEOPLE_THRESHOLD` (default `0.6`: below this the answer is
"Unknown"), `IMGSZ`, `IOU`.

**Limitations:** the classifier only knows the people it was trained on; very similar
signatures or forgeries can fool it. This is a demo (proof of concept), not a tool for legal
or banking identity verification. The signature images are not included in this repository.

## API
| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Server and model status |
| POST | /predict/image | form-data: `file`, `conf` |
| POST | /predict/video | form-data: `file`, `conf` |

## Deployment
- Backend + frontend together: Docker (root `Dockerfile`) on Hugging Face Spaces, Render or Railway.
- Frontend: Netlify, Vercel or GitHub Pages. Set `API_URL` in `frontend/script.js` to the backend URL.
