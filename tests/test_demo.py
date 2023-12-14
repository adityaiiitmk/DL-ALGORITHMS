from testci.trials_ci import capital_case

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'
