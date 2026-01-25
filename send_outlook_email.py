import msal
import requests
import sys

# Configuration - Replace with your values
CLIENT_ID = 'your_client_id_here'
CLIENT_SECRET = 'your_client_secret_here'
TENANT_ID = 'your_tenant_id_here'
SENDER = 'your.email@domain.com'
AUTHORITY = f'https://login.microsoftonline.com/{TENANT_ID}'
SCOPES = ['https://graph.microsoft.com/.default']

def authenticate():
    app = msal.ConfidentialClientApplication(
        CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
    )
    result = app.acquire_token_for_client(scopes=SCOPES)
    if 'access_token' in result:
        return result['access_token']
    else:
        print("Authentication failed:", result.get('error_description'))
        sys.exit(1)

def send_email(access_token, to, subject, body):
    endpoint = f'https://graph.microsoft.com/v1.0/users/{SENDER}/sendMail'
    email_msg = {
        'message': {
            'subject': subject,
            'body': {
                'contentType': 'Text',
                'content': body
            },
            'toRecipients': [
                {
                    'emailAddress': {
                        'address': to
                    }
                }
            ]
        },
        'saveToSentItems': 'true'
    }
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    response = requests.post(endpoint, headers=headers, json=email_msg)
    if response.status_code == 202:
        print("Email sent successfully.")
    else:
        print(f"Failed to send email: {response.status_code} - {response.text}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python send_outlook_email.py <to> <subject> <body>")
        sys.exit(1)
    
    to = sys.argv[1]
    subject = sys.argv[2]
    body = sys.argv[3]
    
    access_token = authenticate()
    send_email(access_token, to, subject, body)