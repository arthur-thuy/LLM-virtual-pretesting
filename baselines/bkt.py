import pandas as pd
from pyBKT.models import Model


def convert_df_for_pybkt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a dataframe to a format that can be used by pyBKT:
    Original columns:
    ['interact_id', 'student_id', 'question_id', 'student_option_id', 'student_option_correct', 'time']
    Output columns (looking at the code within the Model class):
    ['order_id', 'user_id', 'multilearn', 'correct', 'skill_name', 'multiprior', 'multipair', 'multigs', 'folds']
    """
    df = df.sort_values(by=['time'], ascending=True).reset_index().rename(columns={'index': 'order_id'})
    df = df.drop(columns=['interact_id', 'student_option_id', 'time'])
    df = df.rename(columns={
        'student_id': 'user_id',
        'question_id': 'multilearn',
        'student_option_correct': 'correct',
    })
    df['skill_name'] = 'default_skill'
    df['multiprior'] = df['correct']
    df['multipair'] = df['multilearn']
    df['multigs'] = df['multilearn']
    df['folds'] = df['user_id']
    return df


if __name__ == '__main__':
    model = Model(seed=42, num_fits=1)
    # num_fits is just to repeat the run several times to deal with randomness, I believe.
    # Possibly TODO (to improve the baseline): crossvalidation.

    df = convert_df_for_pybkt(pd.read_csv('data/processed/dbe_kt22_interactions_train.csv'))
    model.fit(data=df)
    training_acc = model.evaluate(data=df, metric='accuracy')
    print('training acc:', training_acc)

    df_val = convert_df_for_pybkt(pd.read_csv('data/processed/dbe_kt22_interactions_valsmall.csv'))
    val_acc = model.evaluate(data=df_val, metric='accuracy')
    print('val (small) acc:', val_acc)

    df_val = convert_df_for_pybkt(pd.read_csv('data/processed/dbe_kt22_interactions_vallarge.csv'))
    val_acc = model.evaluate(data=df_val, metric='accuracy')
    print('val (large) acc:', val_acc)

    df_test = convert_df_for_pybkt(pd.read_csv('data/processed/dbe_kt22_interactions_test.csv'))
    test_acc = model.evaluate(data=df_test, metric='accuracy')
    print('test acc:', test_acc)

# Seed 42 / 123 / 1:
    # training acc: 0.8042943275501415
    # val (small) acc: 0.77
    # val (large) acc: 0.84
    # test acc: 0.793
    # Please note this is to predict correct/wrong answer, hence random baseline in 50%, not 25%.
