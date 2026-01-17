"""Payroll helper: compute payroll summary for a single employee.

This module exposes `process_employee_data` used in demos and tests
to compute gross, overtime, bonus, taxes, and net pay for an employee.
"""


def process_employee_data(emp_id, hours_worked, pay_rate, dept):
    """Compute payroll for a single employee and return a summary dict.

    The function handles overtime (time-and-a-half for hours > 40), a small
    department-based bonus, flat tax rate, and returns a consistent result
    dictionary containing gross, tax, and net pay values plus metadata.

    Args:
        emp_id: Employee identifier (string or int).
        hours_worked: Number of hours worked (int, float, or numeric string).
        pay_rate: Hourly pay rate (float).
        dept: Department name (string), case-insensitive.

    Returns:
        dict with keys: id, name, dept, dept_code, gross_pay, overtime_pay,
        bonus, taxes, net_pay, status.
    """
    # Normalize and validate inputs
    emp_id = str(emp_id)
    try:
        hours = float(hours_worked)
    except Exception:
        raise ValueError('hours_worked must be convertible to float')
    pay_rate = float(pay_rate)
    dept = str(dept).strip()

    # Compute regular and overtime hours
    regular_hours = min(hours, 40.0)
    overtime_hours = max(0.0, hours - 40.0)

    regular_pay = regular_hours * pay_rate
    overtime_pay = overtime_hours * pay_rate * 1.5
    gross_pay = regular_pay + overtime_pay

    # Department codes and bonus rates
    dept_map = {
        'sales': ('S', 0.10),
        'it': ('I', 0.05),
        'marketing': ('M', 0.03)
    }
    key = dept.lower()
    dept_code, bonus_rate = dept_map.get(key, ('U', 0.0))
    bonus = gross_pay * bonus_rate

    final_salary = gross_pay + bonus

    # Taxes and net pay
    tax_rate = 0.25
    taxes = final_salary * tax_rate
    net_pay = final_salary - taxes
    if net_pay < 0:
        net_pay = 0.0

    payment_status = 'Due' if final_salary > 0 else 'None'

    employee_name = f"{emp_id} Employee"
    log_entry = f"{employee_name}: gross={gross_pay:.2f}, overtime={overtime_pay:.2f}, bonus={bonus:.2f}, net={net_pay:.2f}\n"

    # Write a brief log entry (append)
    try:
        with open('log.txt', 'a') as fh:
            fh.write(log_entry)
    except Exception:
        # Don't fail the payroll calculation if logging fails
        pass

    result = {
        'id': emp_id,
        'name': employee_name,
        'dept': dept,
        'dept_code': dept_code,
        'gross_pay': round(gross_pay, 2),
        'overtime_pay': round(overtime_pay, 2),
        'bonus': round(bonus, 2),
        'taxes': round(taxes, 2),
        'net_pay': round(net_pay, 2),
        'status': payment_status,
    }

    return result


if __name__ == '__main__':
    # Example usage
    summary = process_employee_data('1001', 40, 15.0, 'Sales')
    print(summary)