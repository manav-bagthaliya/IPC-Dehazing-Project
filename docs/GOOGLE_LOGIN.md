# Turning on "Sign in with Google"

The web app works without any setup, but then **anyone who can open the page can use it**. Follow these steps
(about 5 minutes) to require a Google sign-in. People who sign in for the first time get an account automatically.

You need a free Google Cloud project. Google only gives out the keys below to you, so this part cannot be automated.

## 1. Create the keys

1. Open <https://console.cloud.google.com/> and create (or pick) a project.
2. Go to **APIs & Services → OAuth consent screen**.
   - User type: **External** (or **Internal** if everyone uses the same Google Workspace).
   - Fill in the app name (for example *IPC-Dehaze*) and your email, then save.
   - While the app is in **Testing** mode, add the Google accounts that should be able to sign in under **Test users**.
     To let anyone with a Google account in, publish the app (**Publish app**). The basic scopes used here
     (`openid`, `email`, `profile`) do not need Google's review.
3. Go to **APIs & Services → Credentials → Create credentials → OAuth client ID**.
   - Application type: **Web application**.
   - **Authorised redirect URIs** — add exactly:
     - `http://localhost:5000/auth/google/callback`  (for running on this computer)
     - `https://YOUR-DOMAIN/auth/google/callback`  (only if you deploy it online)
4. Copy the **Client ID** and **Client secret**.

## 2. Give them to the app

Copy `.env.example` to a new file called `.env` in the project folder and fill it in:

```
GOOGLE_CLIENT_ID=1234567890-abcdefg.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-your-secret
```

`.env` is ignored by git, so your secret is not uploaded. **Never share or commit it.**

## 3. Start the app

```powershell
python web/app.py
```

The start-up line should say `Google sign-in: ON`. Open **http://localhost:5000** (use `localhost`, not `127.0.0.1`,
because the redirect address above must match exactly). You will see the sign-in page.

## Optional settings (in `.env`)

| Variable | What it does |
|---|---|
| `ALLOWED_EMAILS` | Comma-separated list. Only these Google accounts may sign in. |
| `ALLOWED_EMAIL_DOMAINS` | Comma-separated list such as `mycompany.com`. Anyone with a matching email may sign in. |
| `PUBLIC_BASE_URL` | Your public address (for example `https://dehaze.example.com`) when deployed behind HTTPS. Turns on secure cookies. |
| `TRUST_PROXY=1` | Set when running behind a reverse proxy such as nginx. |
| `HOST` | Defaults to `127.0.0.1` (this computer only). Use `0.0.0.0` only if you want other devices to connect. |
| `PORT` | Defaults to `5000`. If you change it, change the redirect URI in Google too. |
| `DATA_TTL_HOURS` | How long uploads and results are kept (default 24). |
| `SECRET_KEY` | Optional. If you leave it out, a random key is created once and stored in `instance/secret_key`. |

If both `ALLOWED_EMAILS` and `ALLOWED_EMAIL_DOMAINS` are empty, **any** Google account can sign in.

## Troubleshooting

- **`redirect_uri_mismatch`** — the address in the browser must match the redirect URI in Google exactly
  (scheme, host, port, path). Open the app through `http://localhost:5000`.
- **"Access blocked: app has not completed verification"** — add your account under *Test users*, or publish the app.
- **"This Google account is not allowed"** — the email is not covered by `ALLOWED_EMAILS` / `ALLOWED_EMAIL_DOMAINS`.
- **The app still opens without a sign-in page** — the keys were not found. Check the `.env` file name and location,
  and look for `Google sign-in: ON` in the start-up message.

## How it works (for the curious)

The app uses standard OpenID Connect (the "authorization code" flow) through the Authlib library. After Google confirms
who you are, the app stores only your name, email, picture and Google's stable user id in a signed, HTTP-only session
cookie. Each user's uploads and results live in a separate folder, and the server checks the owner on every request.
All state-changing requests carry a CSRF token. No password is ever handled by this app.
