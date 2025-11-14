def CrisisResponse(user_message: str):
    """
   Crisis response for users flagged as at-risk, with Singapore + International hotlines included.
    """
    response = (
        "I'm really sorry that you're feeling this way. It sounds like you're going through a very difficult time right now, "
        "and I want you to know that you don’t have to face this alone.\n\n"

        "If you’re in **Singapore**, you can reach out to:\n\n"
        "• Samaritans of Singapore (SOS): **1767** (24 hours)\n\n"
        "• Institute of Mental Health (IMH) Helpline: **6389 2222** (24 hours)\n\n"
        "• National Mindline: **1771** (24 hours)\n\n"
        "• Emergency (Police / Ambulance): **999** or **995**\n\n"

        "If you’re **outside Singapore**, please visit [findahelpline.com](https://findahelpline.com) "
        "for international suicide hotlines available in your country.\n\n"

        "If you’re in immediate danger, please go to your nearest emergency department or call your local emergency number.\n\n"
        "Would you like me to share some grounding or breathing exercises while you wait for help?"
    )
    return response
