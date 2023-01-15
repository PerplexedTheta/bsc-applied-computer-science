<!-- login.php - Jake Deery -->
<?php 
include("modules/login.php");

if(isset($_SESSION['sdfd104'])) {
    header("Location: profile.php");
}
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>SDFD104 BY JAKE DEERY | LOGIN</title>
    </head>
    <body>
        <h1>SDFD104 BY JAKE DEERY | LOGIN</h1>
        <h3><?php echo $error; ?></h3>
        <form name="login" action="" method="post">
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
            <hr />
            <p><b>Options: </b><a href="javascript:history.go(-1);">Go back</a> - <a href="./">Return to homepage</a> - <a href="signup.php">Signup</a></p>
        </form>
    </body>
</html>