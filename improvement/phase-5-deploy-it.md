# Phase 5 — Deploy It

**Timeline:** 1 week  
**Goal:** Put your platform on the internet so anyone can use it right now. Understand what "deployment" actually means and why it changes how you build things.

---

## Why This Phase Exists

There is a version of this project that lives entirely on your laptop. You can demo it locally, show it in a screen recording, and describe it in an interview. That version is good.

There is another version that has a live URL. Anyone — a Jeavio interviewer, a potential colleague, your next employer in five years — can open a browser, type a query, and see your system respond in real time. That version is considerably better.

The difference is not just convenience. **Deployment forces you to confront a class of problems that never appear in local development:**

- Secrets management (your API key can't be in the code)
- Dependency pinning (it has to work on a machine you don't control)
- Port binding (your local `localhost:8000` doesn't exist on a server)
- Cold start behaviour (what happens when the server hasn't been used in a while)
- Error visibility (when it breaks in production, how do you know?)

Working through these problems — even once, even on a free tier — teaches you more about production software than weeks of local development.

---

## Concept 1 — What Happens When You Deploy

When you run `uvicorn app.main:app` on your laptop, your computer becomes the server. Requests come in on port 8000, your Python process handles them, and responses go back.

When you deploy to a cloud service like Render, a remote computer (a server somewhere in a data centre) runs that same command. The difference:
- That server is always on (even when your laptop is closed)
- It has a public IP address with a domain name attached to it
- It may have different hardware, a different OS, different available memory

This means your code must be **environment-independent**. It can't assume a specific file path, a specific amount of RAM, or a specific environment variable already being set. This is why Docker exists — to package your code, its dependencies, and its configuration into a single portable unit.

For your first deployment, you don't need Docker. Render's native Python support plus your existing `Procfile` and `requirements.txt` is enough. But understand what they're doing: Render reads your `Procfile` to know how to start your app, and `requirements.txt` to know what to install.

---

## Concept 2 — Why Secrets Can't Be in Your Code

Your `layer5_llm.py` reads `GROQ_API_KEY` from an environment variable. This is correct. Here's why it matters:

If your API key were hardcoded in your source code and you pushed it to GitHub, it would be:
1. Visible to everyone who views the repo (it's public)
2. Permanently in your git history even if you delete it later
3. Potentially scraped by automated bots that scan GitHub for leaked credentials

Groq's free tier has usage limits. If someone scrapes your key, they can exhaust your quota. More importantly, if you ever work with paid APIs (OpenAI, AWS, GCP), a leaked key can result in thousands of dollars in charges.

The correct model is:
- **Development:** store secrets in `.env`, which is in `.gitignore`
- **Production:** set secrets as environment variables in the hosting platform's dashboard (Render has a UI for this)
- **Never:** hardcode secrets in source code or commit them to git

This seems basic but it's one of the most common security mistakes in junior developer portfolios. Having it handled correctly is a positive signal.

---

## Concept 3 — The Cold Start Problem

Many free hosting tiers (Render, Railway, HuggingFace Spaces) will "sleep" your server after a period of inactivity to save resources. When a request comes in and the server is asleep, it has to restart — loading your model, building the ChromaDB index, initialising FastAPI. This can take 20–60 seconds.

This is called a **cold start** and it's a real production concern:
- For your free-tier demo: acceptable, as long as users know to expect it
- For a real application: unacceptable; solved with paid tiers, keepalive pings, or caching

When you present this to Jeavio, acknowledging the cold start behaviour and knowing how to solve it (paid tier with always-on instances, or a cron job that pings the server every 10 minutes) shows production awareness.

Your SBERT model (`all-MiniLM-L6-v2`) is about 80MB. It will be downloaded on first cold start. Add a note in your README about this.

---

## Tasks

---

### 1. Pin your dependencies

**What to do:**  
Your current `requirements.txt` has unpinned dependencies:
```
fastapi
uvicorn
sentence-transformers
```

Replace this with pinned versions:
```
fastapi==0.115.0
uvicorn==0.30.6
sentence-transformers==3.0.1
chromadb==0.5.3
```

Run `pip freeze > requirements.txt` after confirming everything works locally.

**Why this matters:**  
Unpinned dependencies mean "install whatever is latest". That works today. In 6 months, when `sentence-transformers` releases a breaking change, your deployment breaks. Pinned dependencies guarantee that the exact same code runs on every machine, every time.

This is called **reproducible builds** and it's a fundamental DevOps concept. Without it, "it works on my machine" is the only guarantee you have.

---

### 2. Verify your `Procfile`

**What to do:**  
Your `Procfile` should contain:
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The `--host 0.0.0.0` makes the server accept connections from any IP (not just localhost). The `$PORT` lets the hosting platform inject the port it wants your app to listen on. Both are required for cloud deployment.

**Why this matters:**  
`localhost:8000` means "only accept connections from this machine". On a cloud server with no screen, that means "accept connections from nobody". `0.0.0.0` means "accept connections from any IP". This is the single most common reason new developers can't get their apps to work in the cloud.

---

### 3. Deploy to Render

**What to do:**  
1. Create a free account at [render.com](https://render.com)
2. Create a new "Web Service" and connect your GitHub repo
3. Set the runtime to Python 3.11
4. Add your `GROQ_API_KEY` as an environment variable in the Render dashboard
5. Deploy

Render will read your `Procfile`, install from `requirements.txt`, and start your app. The first deploy may take 5–10 minutes while it downloads your SBERT model.

**Why this matters:**  
Render is the fastest path from "local project" to "live URL" for a Python web app. The experience of watching your build logs, finding a dependency that fails to install, fixing it, and redeploying is irreplaceable. You will encounter this same loop dozens of times in your career. Go through it now on a low-stakes project.

Add the live URL to your GitHub README (at the top, prominently) and your LinkedIn profile immediately.

---

### 4. Add a health check endpoint

**What to do:**  
Add a `GET /health` endpoint that returns:
```json
{
  "status": "ok",
  "model_loaded": true,
  "documents_indexed": 30,
  "version": "1.0.0"
}
```

**Why this matters:**  
Health check endpoints are standard in production systems. They're used by:
- Load balancers to decide whether to route traffic to an instance
- Monitoring systems to alert when a service is down
- The deployment platform itself (Render supports health check URLs)

A server that crashes silently and returns 500 errors looks indistinguishable from a slow server to an external observer. A `/health` endpoint tells you in one request whether the server is up *and* whether its dependencies (the model, the index) loaded correctly.

Adding it now is a small effort. Knowing it exists and why is a large signal of engineering maturity.

---

### 5. Record a 2-minute demo

**What to do:**  
Use [Loom](https://loom.com) (free) to record yourself:
1. Opening the live URL in a browser
2. Running a search query and showing the results
3. Running the duplicate detection feature on a document
4. Asking a question through the RAG endpoint

Embed the Loom link in your README at the top.

**Why this matters:**  
A demo video is a force multiplier for your portfolio. It communicates in 2 minutes what a README communicates in 20. Hiring managers who won't read your code will watch a 2-minute demo. A live URL they have to click around to understand is less likely to be engaged with than a video where you show them exactly what to look at.

The other reason: recording yourself explaining your own project is excellent interview preparation. If you can't explain it clearly in a 2-minute video, you can't explain it clearly in an interview.

---

## How You Know Phase 5 Is Complete

- [ ] `requirements.txt` has pinned versions for all dependencies
- [ ] `Procfile` uses `--host 0.0.0.0 --port $PORT`
- [ ] App is live on Render with a public URL
- [ ] Live URL is in the GitHub README at the top
- [ ] `GET /health` endpoint exists and returns model/index status
- [ ] 2-minute Loom demo is embedded in the README

---

## What You Will Have Learned

By the end of Phase 5, you will have deployed a real machine learning application to the cloud and understood the gap between local development and production. You will know what "reproducible builds" mean, why `0.0.0.0` matters, and what a cold start is. These concepts come up in almost every ML engineering role — knowing them from direct experience is worth more than reading about them.
