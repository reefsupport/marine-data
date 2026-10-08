# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| Earlier | No |

## Reporting a vulnerability

Report privately. Do not open a public issue for a vulnerability.

- Use the "Report a vulnerability" button on the
  [Security tab](https://github.com/reefsupport/marine-data/security) of this repository, or
- write to [hello@reef.support](mailto:hello@reef.support).

Include the affected version or commit, what you observed, and steps to reproduce it. Reef Support acknowledges
reports within a few working days and keeps the reporter informed until the issue is resolved.

## What counts as a security report

- A flaw in the `marinedata` package or its tooling.
- Credentials, tokens or keys found in the repository or in the published data.
- Personal data found in the repository or in the published data, such as a recognisable face, a name or an e-mail
  address in an image, label or metadata field.

Licence or attribution problems with a source are data issues, not security reports. Use the
[licence question](https://github.com/reefsupport/marine-data/issues/new?template=licence_question.yml) or
[label error](https://github.com/reefsupport/marine-data/issues/new?template=label_error.yml) form instead. Requests
from rights holders and from people who appear in an image follow the takedown procedure below.

## Takedown requests

Rights holders, the authors of an upstream dataset and people who appear in an image can ask for rows to be removed from
the published datasets, or for a licence or attribution to be corrected. A takedown request is not a vulnerability
report.

### How to ask

- **Public form.** Open a
  [takedown request](https://github.com/reefsupport/marine-data/issues/new?template=takedown_request.yml). Identify the
  rows by `config`, `source` and `image_sha256`. Do not attach or link the image, and do not post personal details.
- **Private route.** Write to [hello@reef.support](mailto:hello@reef.support) if you appear in an image, if the request
  involves personal data, or if you would rather it were not public. Start the subject with "Takedown request". The
  [contact page](https://www.reef.support/contact) works too.

Say what you ask for (remove rows, correct a licence, correct an attribution), which rows are affected, and why you are
entitled to ask, for example that you hold the copyright or that you are the person shown. Reef Support may ask for
further evidence before it acts.

### What happens next

1. Reef Support acknowledges the request within a few working days and says what it will do.
2. If the request is accepted, the affected rows are removed from the published configs in the next dataset revision on
   Hugging Face, and the images are excluded from later builds. A request about a licence or attribution is handled in
   the same revision: the registry entry, the row-level `licence` and `attribution` columns and the dataset card are
   corrected together.
3. The change is noted in [`CHANGELOG.md`](CHANGELOG.md) without naming the requester.
4. Reef Support tells the requester when the revision is live. Copies that others have already downloaded are outside
   its control.
