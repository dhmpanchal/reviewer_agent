from agents.retrieval_agent import RetrievalAgent
from agents.reviewer_agent import ReviewerAgent
from chains.fetch_diagnosis_chain import FetchDiagnosisChain
from utils import Utils

def main():
    retrieval_agent = RetrievalAgent(tag_name="agentic")
    reviewer_agent = ReviewerAgent(tag_name="agentic")
    diagnosis_chain = FetchDiagnosisChain(tag_name="agentic")
    the_utils = Utils()
    
    file_path = "data/pdf/Document 12 Leonard Asthma (1).pdf"
    context_file_path = "data/retrieved_contexts/leonardo_txt_context.txt"
    user_query = """
    Extract a list of all major or chronic medical conditions mentioned in the given patient infromations and need to find dates of the medical conditions from when it detected and calculate the proper dates in formats and from when it was cleaned up or still it is on going.
    """

    # 1. Retrieval Pipeline using Agent tool calls
    patient_info = retrieval_agent.run_retrieval_agent(user_query, file_path)
    the_utils.write_to_file(context_file_path, patient_info)
    # patient_info = the_utils.read_from_file(context_file_path)

    # 2. Diagnosis Pipeline using Task Chain
    result = diagnosis_chain.run_task_chain(patient_info.strip())
    print("======= Task Chain Result =======")
    print(result)
    print("===================================") 

    # result = {
    #     "messages": [
    #         {
    #         "text": "Hi Sarah, According to our records you are a carer. At this difficult time we wanted to let you know that should you need any additional support, please contact us. The practice would be grateful if you"
    #         },
    #         {
    #         "text": "Many Thanks"
    #         },
    #         {
    #         "text": "Patient telephone number (9159.) 07896 234482"
    #         },
    #         {
    #         "text": "Allergic reaction (SN530) "
    #         },
    #         {
    #         "text": "Seen in primary care establishment (XaBET) "
    #         },
    #         {
    #         "text": "Community Pharmacist Consultation Service for minor illness (Y3e4c)"
    #         },
    #         {
    #         "text": "Review a gynae referral. It is unsure if Sarah will need to source a private gynaecologist or nhs due to her previous surgeon being based in Brighton, Previous surgeon has advised that at this point she should be able to be seen by a local gynaecologist."
    #         },
    #         {
    #         "text": "Evorel 100 patches (Theramex HQ UK Ltd) apply two patches twice weekly 16 patch"
    #         },
    #         {
    #         "text": "Evorel 100 patches (Theramex HQ UK Ltd) apply two patches twice weekly"
    #         },
    #         {
    #         "text": "Fexofenadine 120mg tablets take one daily for hayfever symptoms 30 tablet A"
    #         },
    #         {
    #         "text": "Evorel 100 patches (Theramex HQ UK Ltd) apply two patches twice weekly 16 patch"
    #         },
    #         {
    #         "text": "Did not attend (Xa1kG)"
    #         },
    #         {
    #         "text": "Confirmation of your appointment at 08:30 on Fri, 02 of Aug at Cherry Willingham Surgery. If you cannot attend text CANCEL to 07800000199."
    #         },
    #         {
    #         "text": "Appointment Reminder Status: Message Delivery Failed Don\\u2019t forget your appt at 15:50 on Mon 02 of Dec at Cherry Willingham Surgery. If you cannot attend text CANCEL to 07800000199."
    #         }
    #     ],
    #     "appointments": [
    #         {
    #         "date": "Fri, 02 of Aug",
    #         "time": "08:30",
    #         "location": "Cherry Willingham Surgery"
    #         },
    #         {
    #         "date": "Mon, 02 of Dec",
    #         "time": "15:50",
    #         "location": "Cherry Willingham Surgery"
    #         }
    #     ],
    #     "test_results": [
    #         {
    #         "test_name": "Serum prolactin level",
    #         "result": 417,
    #         "unit": "mu/L",
    #         "reference_range": [102, 496]
    #         },
    #         {
    #         "test_name": "Serum oestradiol level",
    #         "result": 219,
    #         "unit": "pmol/L"
    #         },
    #         {
    #         "test_name": "Serum testosterone level",
    #         "result": 0.4,
    #         "unit": "nmol/L",
    #         "reference_range": [0.29, 1.67]
    #         }
    #     ],
    #     "medications": [
    #         {
    #         "medication_name": "Evorel 100 patches",
    #         "dosage": "apply two patches twice weekly"
    #         },
    #         {
    #         "medication_name": "Fexofenadine 120mg tablets",
    #         "dosage": "take one daily for hayfever symptoms"
    #         }
    #     ],
    #     "conditions": [
    #         {
    #         "condition_name": "Prolapse",
    #         "description": "over granulation to an internal element of her vagina, fucibet cream has been applied sporadically when Sarah dilates."
    #         },
    #         {
    #         "condition_name": "Allergic rhinitis due to pollens",
    #         "description": ""
    #         }
    #     ],
    #     "appointments_reminders": [
    #         {
    #         "date": "Fri, 02 of Aug",
    #         "time": "08:30",
    #         "location": "Cherry Willingham Surgery"
    #         },
    #         {
    #         "date": "Mon, 02 of Dec",
    #         "time": "15:50",
    #         "location": "Cherry Willingham Surgery"
    #         }
    #     ],
    #     "test_results_reminders": [
    #         {
    #         "test_name": "Serum prolactin level",
    #         "result": 417,
    #         "unit": "mu/L",
    #         "reference_range": [102, 496]
    #         },
    #         {
    #         "test_name": "Serum oestradiol level",
    #         "result": 219,
    #         "unit": "pmol/L"
    #         },
    #         {
    #         "test_name": "Serum testosterone level",
    #         "result": 0.4,
    #         "unit": "nmol/L",
    #         "reference_range": [0.29, 1.67]
    #         }
    #     ],
    #     "medications_reminders": [
    #         {
    #         "medication_name": "Evorel 100 patches",
    #         "dosage": "apply two patches twice weekly"
    #         },
    #         {
    #         "medication_name": "Fexofenadine 120mg tablets",
    #         "dosage": "take one daily for hayfever symptoms"
    #         }
    #     ],
    #     "conditions_reminders": [
    #         {
    #         "condition_name": "Prolapse",
    #         "description": "over granulation to an internal element of her vagina, fucibet cream has been applied sporadically when Sarah dilates."
    #         },
    #         {
    #         "condition_name": "Allergic rhinitis due to pollens",
    #         "description": ""
    #         }
    #     ]
    # }

    # 3. Reviewer (Reflexion) Agent
    review = reviewer_agent.review(
        task_prompt=diagnosis_chain.prompt_template,
        patient_context=patient_info.strip(),
        task_output=result,
    )
    print("======= Reviewer Comment =======")
    print(review.get("comment", ""))
    print(f"Verdict: {review.get('verdict', '')}")
    print("===================================")

if __name__ == "__main__":
    main()
