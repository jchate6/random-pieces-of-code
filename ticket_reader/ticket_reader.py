"""
Runs through a zendesk json file returning the full message exchange for any given ticket.
Can also return a csv file for each ticket containing ticket number and full exchange.
Date: 6/11/2023
"""
import json
import os
import argparse
import csv


def get_ticket(data, id):
    """
    Retrieve Message data and display to terminal
    :param data: full zendesk ticket dictionary
    :param id: int of zendesk ticket ID
    :return:
    """

    # Loop through ever ticket looking for right one.
    for ticket in data:
        if ticket['id'] == id:
            # Create Header for ticket
            print("=====================================================================")
            print(f"Ticket {ticket['id']}: {ticket['subject']}")
            print(f"From: {ticket['submitter']['name']} ({ticket['submitter']['email']})")
            print(f"Date: {ticket['created_at']}")
            print("---------------------------------------------------------------------")
            # Print Body of Ticket conversation
            print(ticket["description"])
            print("=====================================================================")
            return

    print(f"Sorry, Ticket Number {id} does not exist.")
    return


def save_ticket_text(path, data, csv_filename='lco_ticket_text.csv'):
    """
    Create csv file with Ticket info in ticket directory
    :param path: path to csv file
    :param data: full zendesk ticket dictionary
    :param csv_filename: filename for csv file
    :return:
    """
    header = ['ticket_id', 'ticket_text']
    # Buld csv body
    body = []
    for ticket in data:
        body.append([ticket['id'], ticket["description"]])

    with open(os.path.join(path, csv_filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path and filename for zendesk json file (foo/bar.json)", type=str)
    parser.add_argument("-all", help="Output all data into 'lco_ticket_text.csv' in json_file directory.", default=False, action="store_true")
    args = parser.parse_args()

    # Extact json data from file
    file_path = args.json_file
    data = []
    with open(os.path.join(file_path)) as file:
        # zendesk stores info as a list of dictionaries, so we need to extract each line
        for line in file:
            # convert from json into dictionary
            data.append(json.loads(line))

    # Make csv file and exit if requested
    if args.all:
        save_ticket_text(os.path.dirname(file_path), data)
        quit()

    # Request ticket number and display ticket info
    ticket_number = ''
    while ticket_number != "q":
        ticket_number = input("Enter Ticket Number (q to exit):")
        try:
            get_ticket(data, int(ticket_number))
        except ValueError:
            if ticket_number != "q":
                print("Please enter a valid Ticket Number")
