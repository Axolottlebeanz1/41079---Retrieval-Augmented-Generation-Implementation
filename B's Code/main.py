import time
from retriever import Retriever
from reader import Reader


# ACCESS CONTROL USERS + ROLES

USERS = {
    "admin": {
        "password": "admin123",
        "role": "admin"
    },
    "student": {
        "password": "student123",
        "role": "student"
    },
    "lecturer": {
        "password": "lecturer123",
        "role": "lecturer"
    }
}

# LOGIN SYSTEM

def login():
    print("========== RAG QA System Login ==========")

    attempts = 3

    while attempts > 0:
        username = input("Username: ").strip()
        password = input("Password: ").strip()

        if (
            username in USERS
            and USERS[username]["password"] == password
        ):
            role = USERS[username]["role"]

            print(f"\n✅ Login successful.")
            print(f"Welcome, {username} ({role})\n")

            return username, role

        attempts -= 1
        print(f"❌ Invalid login. Attempts remaining: {attempts}")

    print("\n⛔ Access denied.")
    return None, None


# MAIN SYSTEM

def main():
    while True:
    
        # LOGIN FIRST  

        current_user, current_role = login()

        if current_user is None:
            return
  
        # ROLE-BASED ACCESS    

        if current_role == "student":
            allowed_extensions = [".txt"]

        elif current_role == "lecturer":
            allowed_extensions = [".docx"]

        elif current_role == "admin":
            allowed_extensions = [".txt", ".docx", ".pdf"]

        else:
            allowed_extensions = []

        print(f"📂 Access granted for: {allowed_extensions}")

        retriever = Retriever(
            doc_path="documents",
            allowed_extensions=allowed_extensions
        )

        reader = Reader()
        feedback_log = []
    
        # USER SESSION LOOP
    
        while True:
            print("\nType 'logout' to sign out")
            print("Type 'exit' to close the system")

            query = input("\nWhat is your question?: ").strip()

            # 🔒 Logout current account
            if query.lower() == "logout":
                print(f"\n🔒 {current_user} has been logged out.\n")
                break

            # ❌ Exit entire system
            if query.lower() == "exit":
                print("\n👋 System closed.")
                return
        
            # TIMER START
        
            start_time = time.time()

            docs = retriever.retrieve(query)
            answer = reader.generate_answer(query, docs)

            end_time = time.time()
            processing_time = end_time - start_time

            # OUTPUT

            print(f"\n⏱ Processing Time: {processing_time:.2f} seconds")

            print("\n--- Relevant Documents ---")
            for d in docs:
                if isinstance(d, dict):
                    print(
                        "-",
                        d["snippet"],
                        f"(score: {d['score']:.3f})"
                    )
                else:
                    print("-", d)

            print("\n--- Answer ---")
            print(answer)

            # FEEDBACK SECTION

            while True:
                is_correct = input(
                    "\nWas this answer correct? (yes / no): "
                ).strip().lower()

                if is_correct in ["yes", "no"]:
                    break

                print("Please type only yes or no.")

            while True:
                try:
                    rating = int(input(
                        "Rate correctness (1 = poor, 5 = excellent): "
                    ))

                    if 1 <= rating <= 5:
                        break

                    print("Please enter a number between 1 and 5.")

                except ValueError:
                    print("Please enter a valid number.")

            comment = input(
                "Optional feedback comment (press Enter to skip): "
            ).strip()

            feedback_entry = {
                "user": current_user,
                "role": current_role,
                "question": query,
                "answer": answer,
                "correct": is_correct,
                "rating": rating,
                "comment": comment,
                "processing_time": round(processing_time, 2)
            }

            feedback_log.append(feedback_entry)

            print("\nFeedback saved successfully! ✅")
            print("\n-------------------------------")

        


if __name__ == "__main__":
    main()