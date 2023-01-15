<!-- Modules: login.php - Jake Deery -->
<?php
session_start(); // Starting Session
$error=''; // Variable To Store Error Message
if($_POST){
    // define connection details
    $con = mysqli_connect("127.0.0.1", "username", "passwd", "dbname");

    // check connection
    if (mysqli_connect_errno()) {
        printf("Connect failed: %s\n", mysqli_connect_error());
        exit();
    }

    if (empty($_POST['accountName']) || empty($_POST['password'])) {
            $error = "Username or Password is invalid";
    } else {
        // Define $accountName and $password
        $accountName=$_POST['accountName'];
        $password=$_POST['password'];

        // secure sql
        $accountName = mysqli_real_escape_string($con, $accountName);
        $password = mysqli_real_escape_string($con, $password);

        // query and compare
        $query = mysqli_query($con, "SELECT `accountID` FROM `account` WHERE password='$password' AND accountName='$accountName'");
        $rows = mysqli_num_rows($query);

        // get session_id
        while ($row = mysqli_fetch_assoc($query)) {
            $session_id = $row['accountID'];
        }

        if ($rows == 1) {
            $_SESSION['sdfd104'] = $session_id; // Initializing Session
            header("Location: profile.php"); // Redirecting account page
        } else {
            $error = "Username or Password is invalid";
        }
        mysqli_close($con); // Closing Connection
    }
}
?>
