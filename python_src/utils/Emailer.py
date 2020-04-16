import os
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from configurations.GlobalConstants import output_dir

port = 465  # For SSL
subject = "Your deep learning results are ready for collection"
body = "Whoopsie, these results suck"
smtp_server = 'smtp.gmail.com'
sender_email = 'weaselspythonserver@gmail.com'  # Enter your address
receiver_email = "215029263@student.uj.ac.za"  # Enter receiver address


def open_as_binary_file(filename):
    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    return part


def populate_file_list(directory, qualifier):
    file_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.png') and filename.startswith(qualifier):
            file_list.append(directory + filename)
        elif filename.endswith('.txt') and filename.startswith(qualifier):
            file_list.append(directory + filename)
        else:
            continue
    return file_list


def create_message_with_attachments(filenames):
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Add attachments
    for filename in filenames:
        bin_file = open_as_binary_file(filename)

        encoders.encode_base64(bin_file)
        # Add header as key/value pair to attachment part
        bin_file.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )

        # Add attachment to message
        message.attach(bin_file)

    return message.as_string()


def send_email(text):
    # Create a secure SSL context
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, 'jLcH5sYcELD4Xje')
        server.sendmail(sender_email, receiver_email, text)


def results_dispatch(data_set, architecture):
    file_list = populate_file_list(output_dir, data_set + '_' + architecture)
    message = create_message_with_attachments(file_list)
    send_email(message)

try:
    results_dispatch('ddsm', "vggnet")
except smtplib.SMTPAuthenticationError:
    print('[ERROR] Email credentials could not be authenticated')
