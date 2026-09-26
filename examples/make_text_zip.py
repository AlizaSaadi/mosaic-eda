"""Build examples/datasets/support_tickets.zip: a messy text classification dataset.

Customer support tickets in three folders (billing, shipping, technical), with planted
problems MOSAIC should find:
- exact duplicates (one differs only in case and spacing) and near duplicates
- one ticket filed under two labels (a cross-label duplicate)
- mislabeled tickets (a shipping complaint in billing/, and so on)
- empty and very short tickets
- garbled encoding (UTF-8 read as Windows-1252) and leftover HTML from a web form
- personal data: emails, phone numbers, test card numbers, an SSN-like number
- a confidentiality disclaimer pasted under many tickets (boilerplate)
- tickets in Spanish and French, and one very long ticket (a pasted log dump)

Every value is fake; card numbers are standard test numbers.
Run: python examples/make_text_zip.py
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

OUT = Path(__file__).parent / "datasets" / "support_tickets.zip"
RNG = random.Random(11)

NAMES = [
    "Maya Chen",
    "Omar Haddad",
    "Lena Fischer",
    "Tom Becker",
    "Aisha Khan",
    "Ravi Patel",
    "Sofia Rossi",
    "Jonas Berg",
    "Nina Park",
    "Leo Martin",
    "Grace Liu",
    "Sam Carter",
]
PRODUCTS = ["Acme Cloud", "Priority Plus", "Acme Photo Pro", "Team Workspace", "Acme Backup"]

POOLS = {
    "billing": [
        "I was charged twice for my {product} subscription this month.",
        "My invoice {order} shows an amount of ${amount} but my plan costs less.",
        "Please refund the extra charge on my card as soon as possible.",
        "I cancelled my subscription last week but the payment still went through.",
        "Can you send me a corrected invoice with our company tax number?",
        "The annual plan price increased without any notice to us.",
        "I would like to switch from monthly billing to annual billing.",
        "Your payment page keeps declining my card even though it is valid.",
        "We need a receipt for the payment made on {date} for our accounts team.",
        "The discount code from your newsletter was not applied at checkout.",
        "Why is there a late fee on my account when I paid on time?",
        "I was billed for three seats but we only use two.",
        "Please explain the proration charge on my latest statement.",
        "Our finance department needs invoices sent to a different billing email.",
        "The refund you promised on {date} has not reached my bank yet.",
        "I upgraded to {product} and the charge seems higher than the pricing page.",
    ],
    "shipping": [
        "My package with order {order} has not arrived yet.",
        "The tracking number says delivered but nothing is at my door.",
        "The courier left the parcel at the wrong address on {date}.",
        "Can I change the delivery address for order {order} before it ships?",
        "The box arrived damaged and two items were broken inside.",
        "Delivery was promised within three days and it has been two weeks.",
        "The tracking page has not updated since the parcel left the warehouse.",
        "I received someone else's order instead of mine.",
        "Please send a replacement for the missing item in my shipment.",
        "Is express shipping available to my region for {product} hardware?",
        "The customs fees on this delivery were never mentioned at checkout.",
        "My order was split into two packages and only one has arrived.",
        "The driver marked the package as refused but I never refused it.",
        "Can you arrange a pickup to return the parcel I received today?",
        "The estimated delivery date keeps moving further away.",
        "Our office is closed on weekends, so please deliver on a weekday.",
    ],
    "technical": [
        "The {product} app crashes every time I open the settings page.",
        "I cannot log in after resetting my password this morning.",
        "Sync stopped working after the latest update on {date}.",
        "I get an error 500 when uploading files larger than 2 GB.",
        "Two-factor authentication codes are not arriving on my phone.",
        "The desktop client uses all of my CPU and the fan never stops.",
        "Search results in {product} are missing files I uploaded yesterday.",
        "The browser extension logs me out every few minutes.",
        "After the update the dark mode setting resets on every restart.",
        "Our team cannot share folders because the invite link returns an error.",
        "The API returns a timeout for requests that used to work.",
        "Notifications stopped showing on Android after I installed the new version.",
        "The export to PDF feature produces blank pages.",
        "I see a certificate warning when connecting from our office network.",
        "Backups in {product} fail with the message disk quota exceeded.",
        "The installer hangs at ninety percent on Windows 11.",
    ],
}
OPENERS = ["Hi,", "Hello team,", "Good morning,", "Hello,", "Hi there,", "", "Dear support,"]
CLOSERS = [
    "Thanks",
    "Thank you",
    "Regards, {name}",
    "Best, {name}",
    "Please help asap",
    "Thanks in advance, {name}",
    "Cheers",
]
DISCLAIMER = (
    "CONFIDENTIALITY NOTICE: This email and any attachments are confidential and intended "
    "only for the named recipient."
)
SIGNATURE = "Sent from my iPhone"


def fill(text: str) -> str:
    return text.format(
        product=RNG.choice(PRODUCTS),
        order=f"ORD-{RNG.randint(100000, 999999)}",
        amount=f"{RNG.randint(12, 480)}.{RNG.randint(0, 99):02d}",
        date=f"{RNG.randint(1, 28)} {RNG.choice(['March', 'April', 'May', 'June'])}",
        name=RNG.choice(NAMES),
    )


def ticket(label: str) -> str:
    body = " ".join(fill(s) for s in RNG.sample(POOLS[label], RNG.randint(2, 4)))
    lines = [RNG.choice(OPENERS), body, fill(RNG.choice(CLOSERS))]
    if RNG.random() < 0.4:
        lines.append(DISCLAIMER)
    elif RNG.random() < 0.3:
        lines.append(SIGNATURE)
    return "\n".join(ln for ln in lines if ln) + "\n"


def with_contact(text: str, n: int) -> str:
    name = NAMES[n % len(NAMES)].lower().replace(" ", ".")
    contact = [
        f"You can reach me at {name}@example.com.",
        f"Call me back on +1 415-555-01{n % 90 + 10}.",
        f"My number is (646) 555-02{n % 90 + 10} if email is easier.",
    ][n % 3]
    return text.replace("\n", f" {contact}\n", 1)


def main() -> None:
    sizes = {"billing": 50, "shipping": 44, "technical": 30}
    files: dict[str, str] = {}
    for label, count in sizes.items():
        for i in range(1, count + 1):
            files[f"{label}/ticket_{label[:4]}_{i:03d}.txt"] = ticket(label)

    # personal data in about 1 in 8 tickets, plus card numbers and an SSN-like number
    for n, path in enumerate(sorted(files)[::8]):
        files[path] = with_contact(files[path], n)
    files["billing/ticket_bill_004.txt"] = (
        "Hi,\nPlease stop charging my card 4111 1111 1111 1111, I cancelled in May.\nThanks\n"
    )
    files["billing/ticket_bill_019.txt"] = (
        "Hello team,\nThe charge on card 5555-5555-5555-4444 is wrong, refund it please. "
        "For verification my SSN is 123-45-6789.\nRegards, Leo Martin\n"
    )

    # exact duplicates (one differs only in case and spacing) and near duplicates
    files["billing/ticket_bill_051.txt"] = files["billing/ticket_bill_010.txt"]
    files["shipping/ticket_ship_045.txt"] = files["shipping/ticket_ship_012.txt"]
    files["technical/ticket_tech_031.txt"] = "  " + files["technical/ticket_tech_007.txt"].upper()
    near = files["shipping/ticket_ship_020.txt"]
    files["shipping/ticket_ship_046.txt"] = near.rstrip("\n") + " Still waiting for an update.\n"
    # the same ticket filed under two labels
    files["technical/ticket_tech_032.txt"] = files["billing/ticket_bill_030.txt"]

    # mislabels: complaints filed in the wrong folder
    files["billing/ticket_bill_052.txt"] = (
        "Hello,\nMy package has not arrived and the tracking number says delivered. The courier "
        "left the parcel at the wrong address. Please send a replacement.\nThanks\n"
    )
    files["shipping/ticket_ship_047.txt"] = (
        "Hi,\nThe app crashes every time I log in after the latest update, and sync stopped "
        "working with an error 500.\nRegards, Nina Park\n"
    )
    files["technical/ticket_tech_033.txt"] = (
        "Hello team,\nI was charged twice for my subscription and need a refund of the extra "
        "payment on my invoice.\nThanks\n"
    )

    # empty and very short tickets
    files["billing/ticket_bill_053.txt"] = ""
    files["technical/ticket_tech_034.txt"] = "   \n"
    files["shipping/ticket_ship_048.txt"] = "help??\n"
    files["technical/ticket_tech_035.txt"] = "asap pls\n"

    # garbled encoding: UTF-8 text read as Windows-1252 somewhere upstream
    garbled = [
        "Hi,\nThe café delivery in Zürich was late – again. Could you check the résumé of the "
        "courier's route?\nThanks, Léa\n",
        "Hello,\nMy invoice shows 49,00 € but the price page says 39,00 €. Please explain.\n",
        "Hi there,\nThe app’s “sync now” button doesn’t work since the update.\nCheers\n",
    ]
    for path, text in zip(
        [
            "shipping/ticket_ship_049.txt",
            "billing/ticket_bill_054.txt",
            "technical/ticket_tech_036.txt",
        ],
        garbled,
        strict=True,
    ):
        files[path] = text.encode("utf-8").decode("cp1252", errors="replace")

    # leftover HTML from the web form
    files["technical/ticket_tech_037.txt"] = (
        "<p>Hello,</p><p>The <b>export to PDF</b> feature produces blank pages&nbsp;since "
        "Monday.</p><br><p>Thanks &amp; regards</p>\n"
    )
    files["billing/ticket_bill_055.txt"] = (
        "<div>Hi,<br>I was billed for three seats but we only use two.<br>Please fix the "
        "invoice.</div><div>Grace Liu</div>\n"
    )
    files["shipping/ticket_ship_050.txt"] = (
        "<p>The box arrived <i>damaged</i> &amp; two items were broken inside.</p>"
        "<p>Order ORD-554120</p>\n"
    )

    # other languages
    files["shipping/ticket_ship_051.txt"] = (
        "Hola,\nMi pedido no ha llegado todavía y el número de seguimiento no se actualiza "
        "desde hace una semana. ¿Pueden ayudarme?\nGracias\n"
    )
    files["shipping/ticket_ship_052.txt"] = (
        "Hola,\nEl paquete llegó dañado y faltan dos productos de mi pedido.\nSaludos\n"
    )
    files["billing/ticket_bill_056.txt"] = (
        "Hola,\nMe cobraron dos veces la suscripción este mes, necesito un reembolso por "
        "favor.\nGracias\n"
    )
    files["technical/ticket_tech_038.txt"] = (
        "Bonjour,\nJe ne peux plus me connecter depuis la mise à jour et le mot de passe ne "
        "fonctionne pas.\nMerci\n"
    )
    files["billing/ticket_bill_057.txt"] = (
        "Bonjour,\nLa facture de ce mois est trop élevée et je voudrais un remboursement.\n"
        "Cordialement\n"
    )

    # one very long ticket: a pasted log dump
    log = " ".join(
        f"[{RNG.randint(10, 23)}:{RNG.randint(10, 59)}] worker {RNG.randint(1, 9)} retry "
        f"{RNG.randint(1, 5)} failed with timeout while syncing block {RNG.randint(1000, 9999)}"
        for _ in range(230)
    )
    files["technical/ticket_tech_039.txt"] = (
        "Hi,\nSync keeps failing, here is the full log from the client:\n" + log + "\nThanks\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "support_tickets/README.txt",
            "Synthetic support tickets for MOSAIC EDA. All names, numbers, and cards are fake.\n",
        )
        for path in sorted(files):
            z.writestr(f"support_tickets/{path}", files[path].encode("utf-8"))
    print(f"Wrote {len(files)} tickets to {OUT}")


if __name__ == "__main__":
    main()
