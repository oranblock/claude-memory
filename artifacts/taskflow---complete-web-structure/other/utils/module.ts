import express from 'express';
import { AuthController } from '../controllers/authController';
import { validateRequest } from '../middleware/validateRequest';
import { authenticate } from '../middleware/authenticate';
import { registerSchema, loginSchema } from '../validation/authValidation';

const router = express.Router();
const authController = new AuthController();

/**
 * @route POST /api/auth/register
 * @desc Register a new user
 * @access Public
 */
router.post(
  '/register',
  validateRequest(registerSchema),
  authController.register
);

/**
 * @route POST /api/auth/login
 * @desc Login a user
 * @access Public
 */
router.post(
  '/login',
  validateRequest(loginSchema),
  authController.login
);

/**
 * @route POST /api/auth/logout
 * @desc Logout a user
 * @access Public
 */
router.post('/logout', authController.logout);

/**
 * @route GET /api/auth/current-user
 * @desc Get current user information
 * @access Private
 */
router.get(
  '/current-user',
  authenticate,
  authController.getCurrentUser
);

export default router;