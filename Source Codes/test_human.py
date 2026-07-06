from human_review import requires_human_review

query = "I need a refund for my annual subscription"

if requires_human_review(query):
    print("Human Approval Required")
else:
    print("Auto Approved")