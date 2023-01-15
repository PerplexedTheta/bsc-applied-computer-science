<!-- signup.php - Jake Deery -->
<?php 
include("modules/signup.php");

if(isset($_SESSION['sdfd104'])) {
    header("Location: profile.php");
}
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>SDFD104 BY JAKE DEERY | SIGN UP</title>
    </head>
    <body>
        <h1>SDFD104 BY JAKE DEERY | SIGN UP</h1>
        <form name="signup" action="" method="post">
            <table>
                <tbody>
                    <tr>
                        <td>Account name:</td>
                        <td><input type="text" name="accountName"></td>
                    <tr>
                    <tr>
                        <td>Account password:</td>
                        <td><input type="password" name="password"></td>
                    <tr>
                    <tr>
                        <td></td>
                        <td><input type="submit" value="Submit"></td>
                    </tr>
                </tbody>
            </table>
            <p style="color:red"><b>NOTICE: </b>Passwords are NOT secured! Demo purposes only.</p>
            <hr />
            <p><b>Options: </b><a href="javascript:history.go(-1);">Go back</a> - <a href="./">Return to homepage</a> - <a href="login.php">Login</a></p>
        </form>
    </body>
</html>