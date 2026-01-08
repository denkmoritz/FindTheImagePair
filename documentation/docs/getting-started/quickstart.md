# Quickstart

## Run Backend

(Assumes that dir is `backend/`)

First of all a csv-file with all possible tables has to generated. Simply run:

```bash
python city.py
```

To start the Backend:

```bash
uvicorn main:app --host 0.0.0.0 --reload
```

## Run Frontend

(Assumes that dir is `frontend/`)

```bash
npm run dev
```

The frontend now listens to [http://localhost:5173/](http://localhost:5173/) (use any browser to open it).