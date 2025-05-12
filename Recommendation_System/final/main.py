from data_preprocessing import DataPreprocessor
from recommendation_to_user import JobRecommender

def main(input_file='test.json'):
    """
    Main function to run the job recommendation system.

    Parameters
    ----------
    input_file : str, optional
        Path to input JSON file, by default 'inputJson.json'

    Returns
    -------
    list
        List of recommended job dictionaries
    """
    # Preprocess data
    preprocessor = DataPreprocessor()
    user_df, jobs_df = preprocessor.preprocess(input_file)

    # Generate recommendations
    Recommender = JobRecommender()
    recommendations = Recommender.recommend(user_df, jobs_df)

    # Print recommendations
    # col_to_print = ['title', 'description', 'skills', 'location_type', 'employee_type', 'similarity_score']
    # print("JOB DATA: ")
    # for job in recommendations:
    #     for col in col_to_print:
    #         print(col.upper() + ": " + str(job[col]))  # Access data using user[col]
    #     print("-------------------")
    print(recommendations)

    return recommendations


if __name__ == "__main__":
    main()