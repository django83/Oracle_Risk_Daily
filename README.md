# Oracle Macro-Micro v9.3 — GitHub Actions + email

Pipeline quotidien : collecte des données, détection de régime (BGMM), Early Warning, puis email récapitulatif avec le dashboard HTML en pièce jointe.

## Structure du dépôt

```
oracle_macro.py
requirements.txt
.gitignore
.github/workflows/oracle.yml
```

## Installation

1. Crée un dépôt **privé** sur GitHub et pousse ces fichiers.
2. Dans *Settings → Secrets and variables → Actions → Secrets*, ajoute :

| Secret | Obligatoire | Exemple / remarque |
|---|---|---|
| `FRED_API_KEY` | oui | clé gratuite sur fred.stlouisfed.org |
| `FMP_API_KEY` | non | améliore les séries HYG/IEF/SP500 |
| `SMTP_HOST` | oui | `smtp.gmail.com` |
| `SMTP_PORT` | non | `465` (SSL, défaut) ou `587` (STARTTLS) |
| `SMTP_USER` | oui | ton adresse email |
| `SMTP_PASSWORD` | oui | **mot de passe d'application** (Gmail : compte avec validation en 2 étapes, puis « Mots de passe d'application ») |
| `MAIL_TO` | oui | un ou plusieurs destinataires séparés par des virgules |
| `MAIL_FROM` | non | par défaut = `SMTP_USER` |

3. Optionnel — *Variables* : `MAIL_MODE` = `always` (défaut, un email par jour) ou `alerts` (email seulement si risk-off, changement de régime ou changement de signal).
4. Onglet *Actions → Oracle Macro Daily → Run workflow* pour un premier test. Le premier run fait un calcul complet (plusieurs minutes), les suivants sont incrémentaux.

## Fonctionnement

- **Planification** : lun-ven à 22:30 UTC. GitHub peut retarder les crons de quelques minutes.
- **Persistance** : modèles et historique sont conservés entre les runs via `actions/cache`. Si le cache est perdu (7 jours sans accès) ou si la version de scikit-learn change, le pipeline refait un calcul complet automatiquement.
- **Échec** : en cas d'erreur, un email avec la trace est envoyé et le run est marqué en échec.
- **Dashboard** : joint à l'email et archivé 14 jours dans les artifacts du run.

## Points d'attention

- **Yahoo Finance** limite parfois les IP des runners GitHub. Le script réessaie 4 fois avec pause croissante. Si les échecs sont fréquents, c'est la première chose à regarder (l'email d'erreur le dira).
- **Dépôt inactif** : GitHub désactive les workflows planifiés d'un dépôt sans activité pendant 60 jours. Un simple commit ou un « Run workflow » les réactive.
- **Licence des données** : garde le dépôt privé, et ne commite jamais le dossier `output/`.

## Exécution locale

```bash
pip install -r requirements.txt
export FRED_API_KEY=... SMTP_HOST=... SMTP_USER=... SMTP_PASSWORD=... MAIL_TO=...
python oracle_macro.py
```

Sans variables SMTP, le pipeline tourne normalement et ignore l'envoi d'email.
