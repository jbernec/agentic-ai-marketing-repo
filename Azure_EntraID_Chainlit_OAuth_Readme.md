# Azure Entra ID OAuth Integration for Chainlit Frontend

## Overview
This guide explains how to configure Azure Entra ID (formerly Azure AD) for secure OAuth integration with Chainlit frontend.

## Steps to Configure Azure Entra ID for OAuth with Chainlit

- **Register Application in Entra ID**
  - Go to Azure Portal → Microsoft Entra ID → App registrations → New registration
  - Provide name (e.g., Chainlit-Frontend)
  - Choose Supported account types: Single tenant or Multitenant
  - Add Redirect URI:
    - Production: `https://<your-domain>/auth/oauth/azure-ad/callback`
    - Local Dev: `http://localhost:8000/auth/oauth/azure-ad/callback`

- **Configure Authentication Settings**
  - Enable ID tokens (required for OpenID Connect)
  - Add platform: Web and confirm redirect URI
  - Disable Implicit grant unless needed

- **Set API Permissions**
  - Add Microsoft Graph → Delegated permissions: `openid`, `profile`, `email` (optionally `User.Read`)
  - Click **Grant admin consent** for tenant

- **Create Client Secret**
  - Under Certificates & secrets, create new client secret
  - Copy Client ID, Client Secret, and Tenant ID for later use

- **Configure Environment Variables in Chainlit**
  ```env
  OAUTH_AZURE_AD_CLIENT_ID=<Client ID>
  OAUTH_AZURE_AD_CLIENT_SECRET=<Client Secret>
  OAUTH_AZURE_AD_TENANT_ID=<Tenant ID>
  CHAINLIT_URL=<your-domain>
  CHAINLIT_AUTH_SECRET=<secure-random-string>
  OAUTH_AZURE_AD_ENABLE_SINGLE_TENANT=true
  ```

- **Verify Scopes**
  - Ensure `openid` scope included in OAuth request

- **Test Flow**
  - Restart Chainlit app
  - Click **Login with Microsoft** and confirm redirect/token exchange

## Best Practices
- Use HTTPS for production redirect URIs
- Apply Conditional Access Policies for compliance
- Store secrets in Azure Key Vault
- For modular deployments (frontend vs agent), register separate apps for each component

![OAuth Flow Diagram](oauth_flow_chainlit_entraid.png)
